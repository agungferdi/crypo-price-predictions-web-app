import * as tf from '@tensorflow/tfjs';

export interface ModelMetrics {
  mse: number;  // Mean Squared Error
  mae: number;  // Mean Absolute Error
  rmse: number; // Root Mean Squared Error
}

export class PriceModel {
  private model: tf.Sequential | null = null;
  private lookback: number = 30; // Number of days to look back for prediction
  private isModelReady: boolean = false;

  /**
   * Initialize the TensorFlow.js model
   */
  async init(): Promise<void> {
    await tf.ready();
    this.model = tf.sequential();
    
    // Add layers to the model
    // LSTM layer with 50 units and return sequences
    this.model.add(tf.layers.lstm({
      units: 50,
      returnSequences: true,
      inputShape: [this.lookback, 1]
    }));
    
    // Add dropout for regularization
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    
    // Another LSTM layer
    this.model.add(tf.layers.lstm({
      units: 50,
      returnSequences: false
    }));
    
    // Output layer
    this.model.add(tf.layers.dense({ units: 1 }));
    
    // Compile the model
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mse', 'mae']
    });
    
    this.isModelReady = true;
  }

  /**
   * Train the model with historical price data
   */
  async train(data: number[][], epochs: number = 20): Promise<tf.History> {
    if (!this.isModelReady || !this.model) {
      await this.init();
    }
    
    // Prepare data for training
    const { inputs, targets } = this.prepareData(data);
    
    // Train the model
    const history = await this.model!.fit(inputs, targets, {
      epochs,
      batchSize: 32,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 })
    });
    
    // Clean up tensors to prevent memory leaks
    inputs.dispose();
    targets.dispose();
    
    return history;
  }

  /**
   * Make predictions for future prices
   */
  async predict(data: number[][], days: number): Promise<number[]> {
    if (!this.isModelReady || !this.model) {
      throw new Error('Model not ready. Please initialize and train first.');
    }
    
    const prices = data.map(d => d[1]);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    
    // Get last sequence for prediction
    const lastSequence = prices.slice(-this.lookback).map(p => (p - min) / (max - min));
    let currentSequence = [...lastSequence];
    
    const predictions: number[] = [];
    
    // Generate predictions day by day
    for (let i = 0; i < days; i++) {
      // Reshape for model input
      const inputTensor = tf.tensor3d([currentSequence], [1, this.lookback, 1]);
      
      // Predict next value
      const predictionTensor = this.model!.predict(inputTensor) as tf.Tensor;
      const predictionValue = predictionTensor.dataSync()[0];
      
      // Denormalize to get actual price
      const actualPrediction = predictionValue * (max - min) + min;
      predictions.push(actualPrediction);
      
      // Update sequence for next prediction
      currentSequence.shift();
      currentSequence.push(predictionValue);
      
      // Clean up tensors
      tf.dispose(inputTensor);
      tf.dispose(predictionTensor);
    }
    
    return predictions;
  }

  /**
   * Calculate model performance metrics
   */
  evaluateModel(actualPrices: number[], predictedPrices: number[]): ModelMetrics {
    const length = Math.min(actualPrices.length, predictedPrices.length);
    let sumSquaredError = 0;
    let sumAbsoluteError = 0;
    
    for (let i = 0; i < length; i++) {
      const error = actualPrices[i] - predictedPrices[i];
      sumSquaredError += error * error;
      sumAbsoluteError += Math.abs(error);
    }
    
    const mse = sumSquaredError / length;
    const mae = sumAbsoluteError / length;
    const rmse = Math.sqrt(mse);
    
    return { mse, mae, rmse };
  }

  /**
   * Prepare data for the model
   */
  private prepareData(data: number[][]): { inputs: tf.Tensor3D; targets: tf.Tensor2D } {
    // Extract prices from data
    const prices = data.map(d => d[1]);
    
    // Normalize data between 0-1
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const normalizedPrices = prices.map(p => (p - min) / (max - min));
    
    // Create sequences for training
    const sequences: number[][] = [];
    const targets: number[] = [];
    
    for (let i = 0; i < normalizedPrices.length - this.lookback; i++) {
      sequences.push(normalizedPrices.slice(i, i + this.lookback));
      targets.push(normalizedPrices[i + this.lookback]);
    }
    
    // Convert to tensors
    const inputTensor = tf.tensor3d(sequences, [sequences.length, this.lookback, 1]);
    const targetTensor = tf.tensor2d(targets, [targets.length, 1]);
    
    return {
      inputs: inputTensor,
      targets: targetTensor
    };
  }
  
  /**
   * Dispose of the model and free up memory
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isModelReady = false;
    }
  }
}

// Create a singleton instance
export const priceModel = new PriceModel();