import * as tf from '@tensorflow/tfjs';

export interface ModelMetrics {
  mse: number;
  mae: number;
  rmse: number;
}

export class PriceModel {
  private model: tf.Sequential | null = null;
  private lookback: number = 30;
  private isModelReady: boolean = false;

  async init(): Promise<void> {
    await tf.ready();
    this.model = tf.sequential();
    
    this.model.add(tf.layers.lstm({
      units: 50,
      returnSequences: true,
      inputShape: [this.lookback, 1]
    }));
    
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    
    this.model.add(tf.layers.lstm({
      units: 50,
      returnSequences: false
    }));
    
    this.model.add(tf.layers.dense({ units: 1 }));
    
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mse', 'mae']
    });
    
    this.isModelReady = true;
  }

  async train(data: number[][], epochs: number = 20): Promise<tf.History> {
    if (!this.isModelReady || !this.model) {
      await this.init();
    }
    
    const { inputs, targets } = this.prepareData(data);
    
    const history = await this.model!.fit(inputs, targets, {
      epochs,
      batchSize: 32,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 })
    });
    
    inputs.dispose();
    targets.dispose();
    
    return history;
  }

  async predict(data: number[][], days: number): Promise<number[]> {
    if (!this.isModelReady || !this.model) {
      throw new Error('Model not ready. Please initialize and train first.');
    }
    
    const prices = data.map(d => d[1]);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    
    const lastSequence = prices.slice(-this.lookback).map(p => (p - min) / (max - min));
    let currentSequence = [...lastSequence];
    
    const predictions: number[] = [];
    
    for (let i = 0; i < days; i++) {
      const reshapedSequence: number[][][] = [currentSequence.map(value => [value])];
      const inputTensor = tf.tensor3d(reshapedSequence);
      
      const predictionTensor = this.model!.predict(inputTensor) as tf.Tensor;
      const predictionValue = predictionTensor.dataSync()[0];
      
      const actualPrediction = predictionValue * (max - min) + min;
      predictions.push(actualPrediction);
      
      currentSequence.shift();
      currentSequence.push(predictionValue);
      
      tf.dispose(inputTensor);
      tf.dispose(predictionTensor);
    }
    
    return predictions;
  }

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

  private prepareData(data: number[][]): { inputs: tf.Tensor3D; targets: tf.Tensor2D } {
    const prices = data.map(d => d[1]);
    
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const normalizedPrices = prices.map(p => (p - min) / (max - min));
    
    const sequencesRaw: number[][] = [];
    const targets: number[] = [];
    
    for (let i = 0; i < normalizedPrices.length - this.lookback; i++) {
      sequencesRaw.push(normalizedPrices.slice(i, i + this.lookback));
      targets.push(normalizedPrices[i + this.lookback]);
    }
    
    const sequences: number[][][] = sequencesRaw.map(seq => seq.map(value => [value]));
    
    const inputTensor = tf.tensor3d(sequences);
    const targetTensor = tf.tensor2d(targets, [targets.length, 1]);
    
    return {
      inputs: inputTensor,
      targets: targetTensor
    };
  }
  
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isModelReady = false;
    }
  }
}

export const priceModel = new PriceModel();