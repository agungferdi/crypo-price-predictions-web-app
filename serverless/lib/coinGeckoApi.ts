import axios from 'axios';

// Types for the API responses
export interface ChartData {
  prices: [number, number][];
  market_caps: [number, number][];
  total_volumes: [number, number][];
}

export interface CoinMarketData {
  id: string;
  market_data: {
    current_price: Record<string, number>;
    price_change_percentage_24h: number;
    price_change_percentage_7d: number;
    price_change_percentage_30d: number;
    price_change_percentage_60d: number;
    price_change_percentage_200d: number;
    price_change_percentage_1y: number;
    market_cap: Record<string, number>;
    total_volume: Record<string, number>;
  };
}

class CoinGeckoAPI {
  private readonly API_BASE_URL = 'https://api.coingecko.com/api/v3';
  private readonly API_KEY: string;

  constructor() {
    this.API_KEY = process.env.COINGECKO_API_KEY || 'CG-sou5TJWDNdLnDCbWMTPyi6bT';
  }

  private createHeaders() {
    return {
      'x-cg-demo-api-key': this.API_KEY,
      'Content-Type': 'application/json',
    };
  }

  // Get historical price data for a coin
  async getCoinHistoricalData(
    coinId: string,
    days: number = 365,
    currency: string = 'usd'
  ): Promise<ChartData> {
    try {
      const response = await axios.get(
        `${this.API_BASE_URL}/coins/${coinId}/market_chart`,
        {
          params: {
            vs_currency: currency,
            days: days,
            interval: 'daily',
          },
          headers: this.createHeaders(),
        }
      );

      return response.data;
    } catch (error) {
      console.error(`Error fetching historical data for ${coinId}:`, error);
      throw error;
    }
  }

  // Get detailed market data for a coin
  async getCoinMarketData(
    coinId: string,
    currency: string = 'usd'
  ): Promise<CoinMarketData> {
    try {
      const response = await axios.get(
        `${this.API_BASE_URL}/coins/${coinId}`,
        {
          params: {
            localization: false,
            tickers: false,
            market_data: true,
            community_data: false,
            developer_data: false,
            sparkline: false,
          },
          headers: this.createHeaders(),
        }
      );

      return response.data;
    } catch (error) {
      console.error(`Error fetching market data for ${coinId}:`, error);
      throw error;
    }
  }
}

// Export a singleton instance
export const coinGeckoAPI = new CoinGeckoAPI();