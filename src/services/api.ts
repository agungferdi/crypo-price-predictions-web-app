import axios from 'axios';

const API_KEY = 'CG-sou5TJWDNdLnDCbWMTPyi6bT';
const BASE_URL = 'https://api.coingecko.com/api/v3';
// CORS proxy for production environment
const CORS_PROXY = 'https://corsproxy.io/?';
const IS_PRODUCTION = process.env.NODE_ENV === 'production' || window.location.hostname !== 'localhost';

// Use CORS proxy in production, direct API in development
const getBaseUrl = () => {
  return IS_PRODUCTION ? `${CORS_PROXY}${encodeURIComponent(BASE_URL)}` : BASE_URL;
};

const api = axios.create({
  baseURL: getBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
    'x-cg-demo-api-key': API_KEY,
  },
  timeout: 15000,
});

export interface Coin {
  id: string;
  symbol: string;
  name: string;
  image: string;
  current_price: number;
  market_cap: number;
  market_cap_rank: number;
  price_change_percentage_24h: number;
  price_change_percentage_7d_in_currency?: number;
  price_change_percentage_30d_in_currency?: number;
  price_change_percentage_1y_in_currency?: number;
  total_volume: number;
  circulating_supply?: number;
  max_supply?: number;
  ath?: number;
  ath_change_percentage?: number;
}

export interface ChartData {
  prices: [number, number][];
  market_caps: [number, number][];
  total_volumes: [number, number][];
}

export interface CoinDetail {
  id: string;
  symbol: string;
  name: string;
  market_data: {
    current_price: {
      [key: string]: number;
    };
    price_change_percentage_24h: number;
    price_change_percentage_7d: number;
    price_change_percentage_30d: number;
    price_change_percentage_1y: number;
    market_cap: {
      [key: string]: number;
    };
    total_volume: {
      [key: string]: number;
    };
  };
  image: {
    small: string;
    large: string;
  };
  description: {
    en: string;
  };
  links: {
    homepage: string[];
    blockchain_site: string[];
  };
}

export const coinGeckoAPI = {
  ping: async () => {
    try {
      const response = await api.get('/ping');
      return response.data;
    } catch (error) {
      console.error('Error pinging CoinGecko API:', error);
      throw error;
    }
  },

  getCoins: async () => {
    try {
      const response = await api.get('/coins/list');
      return response.data;
    } catch (error) {
      console.error('Error fetching coin list:', error);
      throw error;
    }
  },

  getMarkets: async (currency = 'usd', page = 1, perPage = 20) => {
    try {
      const response = await api.get('/coins/markets', {
        params: {
          vs_currency: currency,
          order: 'market_cap_desc',
          per_page: perPage,
          page: page,
          sparkline: false,
          price_change_percentage: '24h,7d,30d,1y',
        },
      });
      return response.data as Coin[];
    } catch (error) {
      console.error('Error fetching coin markets:', error);
      throw error;
    }
  },

  getCoinDetails: async (coinId: string) => {
    try {
      const response = await api.get(`/coins/${coinId}`, {
        params: {
          localization: false,
          tickers: false,
          market_data: true,
          community_data: false,
          developer_data: false,
          sparkline: false,
        },
      });
      return response.data as CoinDetail;
    } catch (error) {
      console.error(`Error fetching details for coin ${coinId}:`, error);
      throw error;
    }
  },

  getPrices: async (coinIds: string[], currencies = ['usd']) => {
    try {
      const response = await api.get('/simple/price', {
        params: {
          ids: coinIds.join(','),
          vs_currencies: currencies.join(','),
          include_24hr_change: true,
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching coin prices:', error);
      throw error;
    }
  },

  getCoinChart: async (coinId: string, days = 7, currency = 'usd') => {
    try {
      const response = await api.get(`/coins/${coinId}/market_chart`, {
        params: {
          vs_currency: currency,
          days: days,
        },
      });
      return response.data as ChartData;
    } catch (error) {
      console.error(`Error fetching chart data for coin ${coinId}:`, error);
      throw error;
    }
  },

  getCoinChartRange: async (coinId: string, from: number, to: number, currency = 'usd') => {
    try {
      const response = await api.get(`/coins/${coinId}/market_chart/range`, {
        params: {
          vs_currency: currency,
          from: from,
          to: to,
        },
      });
      return response.data as ChartData;
    } catch (error) {
      console.error(`Error fetching chart range data for coin ${coinId}:`, error);
      throw error;
    }
  },

  searchCoins: async (query: string) => {
    try {
      const response = await api.get('/search', {
        params: {
          query,
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error searching coins:', error);
      throw error;
    }
  },

  getTrendingCoins: async () => {
    try {
      const response = await api.get('/search/trending');
      return response.data;
    } catch (error) {
      console.error('Error fetching trending coins:', error);
      throw error;
    }
  },
};