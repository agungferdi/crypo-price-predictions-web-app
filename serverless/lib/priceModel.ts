import * as tf from '@tensorflow/tfjs-node';
import { ChartData } from './coinGeckoApi';

// Timeframes for predictions
export type TimeFrame = '1d' | '7d' | '30d' | '365d';

// Model metrics interface
export interface ModelMetrics {
  mse: number;
  mae: number;
  rmse: number;
}

// Prediction result interface
export interface PredictionResult {
  predictions: Record<TimeFrame, number>;
  confidence: number;
  metrics: ModelMetrics;
}

export class PriceModel {
  // Feature engineering: Create additional features from price data
  private engineerFeatures(priceData: number[], volumeData: number[]) {
    const features = [];
    const windowSizes = [5, 10, 20]; // Different window sizes for moving averages
    
    for (let i = Math.max(...windowSizes); i < priceData.length; i++) {
      const featureSet = [];
      
      // Raw price
      featureSet.push(priceData[i]);
      
      // Price changes (returns)
      featureSet.push(priceData[i] - priceData[i-1]);
      featureSet.push((priceData[i] - priceData[i-1]) / priceData[i-1]);
      
      // Volume
      featureSet.push(volumeData[i]);
      featureSet.push((volumeData[i] - volumeData[i-1]) / volumeData[i-1]);
      
      // Moving averages
      for (const windowSize of windowSizes) {
        const window = priceData.slice(i - windowSize, i);
        const ma = window.reduce((sum, price) => sum + price, 0) / windowSize;
        featureSet.push(ma);
        
        // Price relative to moving average
        featureSet.push(priceData[i] / ma - 1);
      }
      
      // Volatility (standard deviation over windows)
      for (const windowSize of windowSizes) {
        const window = priceData.slice(i - windowSize, i);
        const mean = window.reduce((sum, price) => sum + price, 0) / windowSize;
        const variance = window.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / windowSize;
        featureSet.push(Math.sqrt(variance));
      }
      
      // Add more advanced technical indicators here
      // RSI (Relative Strength Index) - simplified version
      if (i >= 14) {
        const gains = [];
        const losses = [];
        for (let j = i - 14; j < i; j++) {
          const change = priceData[j] - priceData[j-1];
          if (change >= 0) {
            gains.push(change);
            losses.push(0);
          } else {
            gains.push(0);
            losses.push(Math.abs(change));
          }
        }
        
        const avgGain = gains.reduce((sum, val) => sum + val, 0) / 14;
        const avgLoss = losses.reduce((sum, val) => sum + val, 0) / 14;
        
        if (avgLoss !== 0) {
          const rs = avgGain / avgLoss;
          const rsi = 100 - (100 / (1 + rs));
          featureSet.push(rsi / 100); // Normalize to 0-1
        } else {
          featureSet.push(1); // If no losses, RSI is 100
        }
      } else {
        featureSet.push(0.5); // Default value when insufficient data
      }
      
      features.push(featureSet);
    }
    
    return features;
  }

  // Calculate metrics for model evaluation
  private calculateMetrics(actual: number[], predicted: number[]): ModelMetrics {
    const n = actual.length;
    let sumSquaredError = 0;
    let sumAbsError = 0;
    
    for (let i = 0; i < n; i++) {
      const error = actual[i] - predicted[i];
      sumSquaredError += error * error;
      sumAbsError += Math.abs(error);
    }
    
    const mse = sumSquaredError / n;
    const mae = sumAbsError / n;
    const rmse = Math.sqrt(mse);
    
    return { mse, mae, rmse };
  }

  // Train model and make predictions
  async trainAndPredict(data: ChartData): Promise<PredictionResult> {
    if (!data || !data.prices || data.prices.length < 60) {
      throw new Error('Insufficient historical data for prediction');
    }
    
    try {
      // Extract and prepare data
      const prices = data.prices.map(dataPoint => dataPoint[1]);
      const volumes = data.total_volumes.map(dataPoint => dataPoint[1]);
      const dates = data.prices.map(dataPoint => dataPoint[0]);
      
      // Normalize data
      const minPrice = Math.min(...prices);
      const maxPrice = Math.max(...prices);
      const normalizedPrices = prices.map(p => (p - minPrice) / (maxPrice - minPrice));
      
      const minVolume = Math.min(...volumes);
      const maxVolume = Math.max(...volumes);
      const normalizedVolumes = volumes.map(v => (v - minVolume) / (maxVolume - minVolume));

      // Create sequences for time series prediction
      const lookBack = 30; // Look at 30 days of data to predict
      
      // Create input sequences and labels
      const X = [];
      const y = [];
      
      for (let i = lookBack; i < normalizedPrices.length; i++) {
        // Create a sequence with both price and volume data
        const sequence = [];
        for (let j = i - lookBack; j < i; j++) {
          sequence.push([
            normalizedPrices[j],
            normalizedVolumes[j]
          ]);
        }
        X.push(sequence);
        y.push(normalizedPrices[i]);
      }

      // Split data into training and validation sets (80/20)
      const splitIdx = Math.floor(X.length * 0.8);
      
      const X_train = X.slice(0, splitIdx);
      const y_train = y.slice(0, splitIdx);
      
      const X_val = X.slice(splitIdx);
      const y_val = y.slice(splitIdx);
      
      // Convert to tensors
      const xsTrain = tf.tensor3d(X_train);
      const ysTrain = tf.tensor2d(y_train.map(val => [val]));
      
      const xsVal = tf.tensor3d(X_val);
      const ysVal = tf.tensor2d(y_val.map(val => [val]));
      
      // Build improved model with multiple inputs
      const model = tf.sequential();
      
      // LSTM layers for time series
      model.add(tf.layers.lstm({
        units: 100, 
        returnSequences: true,
        inputShape: [lookBack, 2] // [timesteps, features] - price and volume
      }));
      
      model.add(tf.layers.dropout({ rate: 0.2 }));
      
      model.add(tf.layers.lstm({
        units: 50,
        returnSequences: false
      }));
      
      model.add(tf.layers.dropout({ rate: 0.2 }));
      
      model.add(tf.layers.dense({ units: 25, activation: 'relu' }));
      model.add(tf.layers.dense({ units: 1 }));
      
      // Compile the model
      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError',
        metrics: ['mae']
      });
      
      // Train the model with early stopping
      await model.fit(xsTrain, ysTrain, {
        epochs: 100,
        batchSize: 32,
        validationData: [xsVal, ysVal],
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (epoch > 20 && logs && logs.val_loss > 0.02) {
              model.stopTraining = true;
            }
          }
        }
      });
      
      // Evaluate model on validation set
      const valPredictions = model.predict(xsVal) as tf.Tensor;
      const valPredArray = await valPredictions.dataSync();
      const valActArray = await ysVal.dataSync();
      
      // Calculate metrics
      const predictionMetrics = this.calculateMetrics(
        Array.from(valActArray), 
        Array.from(valPredArray)
      );
      
      // Make predictions for multiple timeframes
      const timeFrames: TimeFrame[] = ['1d', '7d', '30d', '365d'];
      const predictions: Record<TimeFrame, number> = {} as Record<TimeFrame, number>;
      
      // Get latest sequence for prediction
      const lastSequence = [];
      for (let i = normalizedPrices.length - lookBack; i < normalizedPrices.length; i++) {
        lastSequence.push([
          normalizedPrices[i],
          normalizedVolumes[i]
        ]);
      }
      
      // Make predictions for each timeframe
      for (const timeFrame of timeFrames) {
        let daysAhead = 1;
        if (timeFrame === '7d') daysAhead = 7;
        if (timeFrame === '30d') daysAhead = 30;
        if (timeFrame === '365d') daysAhead = 365;
        
        let currentPrediction = [...lastSequence];
        
        // Predict iteratively for the required days
        for (let day = 0; day < daysAhead; day++) {
          const predictSequence = currentPrediction.slice(currentPrediction.length - lookBack);
          
          // Reshape for prediction
          const inputTensor = tf.tensor3d([predictSequence]);
          
          // Predict
          const predResult = model.predict(inputTensor) as tf.Tensor;
          const predValue = predResult.dataSync()[0];
          
          // For volume prediction (simplified), use the last known volume
          const lastVolume = currentPrediction[currentPrediction.length - 1][1];
          
          // Add prediction to current data for next iteration
          currentPrediction.push([predValue, lastVolume]);
          
          // Clean up tensors
          inputTensor.dispose();
          predResult.dispose();
        }
        
        // Get the last value (our prediction)
        const finalPredValue = currentPrediction[currentPrediction.length - 1][0];
        
        // Denormalize
        predictions[timeFrame] = finalPredValue * (maxPrice - minPrice) + minPrice;
      }
      
      // Calculate confidence from validation metrics
      const priceRange = maxPrice - minPrice;
      const normalizedRMSE = predictionMetrics.rmse * priceRange; // Convert to original scale
      const relativeError = normalizedRMSE / (prices[prices.length - 1]); // Error relative to current price
      
      // Convert to confidence percentage (inverse of relative error)
      let confidenceScore = 100 * (1 - Math.min(1, relativeError * 10));
      
      // Adjust for dataset size
      const dataFactor = Math.min(1, data.prices.length / 180);
      confidenceScore = confidenceScore * (0.7 + 0.3 * dataFactor);
      
      // Cap confidence
      const confidence = Math.max(10, Math.min(95, confidenceScore));
      
      // Clean up tensors
      xsTrain.dispose();
      ysTrain.dispose();
      xsVal.dispose();
      ysVal.dispose();
      model.dispose();
      
      return {
        predictions,
        confidence,
        metrics: predictionMetrics
      };
      
    } catch (err) {
      console.error('Error in price prediction model:', err);
      throw new Error('Failed to train prediction model');
    }
  }
}