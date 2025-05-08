import { coinGeckoAPI, ChartData } from './api';
import { PriceModel } from './priceModel';

export type TimeFrame = '1d' | '7d' | '30d' | '365d' | '2y' | '4y';

export interface ModelMetrics {
  mse: number;
  mae: number;
  rmse: number;
}

export interface ForecastResult {
  predictions: Record<TimeFrame, number>;
  confidence: number;
  metrics: ModelMetrics;
  currentPrice: number;
  changePercentages: Record<TimeFrame, string>;
  currency: string;
}

type ModelProgressCallback = (step: string, epoch: number, totalEpochs: number, logs?: any) => void;

export const forecastsAPI = {
  isApiAvailable: async (): Promise<boolean> => {
    try {
      if (typeof window !== 'undefined') {
        await import('@tensorflow/tfjs');
        return true;
      }
      return false;
    } catch (e) {
      console.error('TensorFlow.js is not available:', e);
      return false;
    }
  },

  getPredictions: async (
    coinId: string, 
    currency: string,
    days: number = 365,
    onProgress?: ModelProgressCallback
  ): Promise<ForecastResult> => {
    try {
      if (onProgress) onProgress('loading', 0, 100);
      
      const chartData = await coinGeckoAPI.getCoinChart(coinId, days, currency);
      
      if (!chartData || !chartData.prices || chartData.prices.length < 30) {
        throw new Error('Insufficient historical data for forecasting');
      }

      if (onProgress) onProgress('preparing', 0, 100);
      
      const model = new PriceModel();
      
      const prices = chartData.prices.map(p => p[1]);
      const timestamps = chartData.prices.map(p => p[0]);
      
      await model.train(
        prices, 
        timestamps,
        (epoch, totalEpochs, logs) => {
          if (onProgress) onProgress('training', epoch, totalEpochs, logs);
        }
      );
      
      if (onProgress) onProgress('predicting', 0, 100);
      
      const currentPrice = prices[prices.length - 1];
      
      const predictionDays = {
        '1d': 1,
        '7d': 7, 
        '30d': 30,
        '365d': 365,
        '2y': 365 * 2,
        '4y': 365 * 4
      };
      
      const predictions: Record<TimeFrame, number> = {
        '1d': 0,
        '7d': 0,
        '30d': 0,
        '365d': 0,
        '2y': 0,
        '4y': 0
      };
      
      for (const [timeframe, days] of Object.entries(predictionDays)) {
        predictions[timeframe as TimeFrame] = await model.predict(prices, days);
      }
      
      const metrics = model.getMetrics();
      
      const confidence = Math.max(0, Math.min(100, 100 - (metrics.rmse / currentPrice) * 100));
      
      const changePercentages: Record<TimeFrame, string> = {
        '1d': '',
        '7d': '',
        '30d': '',
        '365d': '',
        '2y': '',
        '4y': ''
      };
      
      Object.keys(predictions).forEach(timeframe => {
        const tf = timeframe as TimeFrame;
        const percentChange = ((predictions[tf] - currentPrice) / currentPrice) * 100;
        const sign = percentChange >= 0 ? '+' : '';
        changePercentages[tf] = `${sign}${percentChange.toFixed(2)}%`;
      });
      
      if (onProgress) onProgress('predicting', 100, 100);
      
      model.dispose();
      
      return {
        predictions,
        confidence,
        metrics,
        currentPrice,
        changePercentages,
        currency
      };
      
    } catch (error) {
      console.error('Error making price predictions:', error);
      throw error;
    }
  }
};