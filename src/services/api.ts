import axios, { AxiosError } from 'axios';

const API_KEY = 'CG-sou5TJWDNdLnDCbWMTPyi6bT';
const BASE_URL = 'https://api.coingecko.com/api/v3';

// Create axios instance with default config
const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'x-cg-demo-api-key': API_KEY, // Use headers for authentication (recommended approach)
  },
  timeout: 15000, // 15 second timeout
});

// Sample data for fallback when API fails
const fallbackCoins: Coin[] = [
  {
    id: 'bitcoin',
    symbol: 'btc',
    name: 'Bitcoin',
    image: 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
    current_price: 62341.12,
    market_cap: 1224538672341,
    market_cap_rank: 1,
    price_change_percentage_24h: 2.35,
    price_change_percentage_7d_in_currency: 5.78,
    price_change_percentage_30d_in_currency: -3.21,
    price_change_percentage_1y_in_currency: 42.56,
    total_volume: 38726543210,
    circulating_supply: 19356250,
    max_supply: 21000000,
  },
  {
    id: 'ethereum',
    symbol: 'eth',
    name: 'Ethereum',
    image: 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
    current_price: 3021.67,
    market_cap: 362847293641,
    market_cap_rank: 2,
    price_change_percentage_24h: 1.62,
    price_change_percentage_7d_in_currency: 4.12,
    price_change_percentage_30d_in_currency: -2.45,
    price_change_percentage_1y_in_currency: 35.12,
    total_volume: 21543876231,
    circulating_supply: 120234500,
  },
  {
    id: 'tether',
    symbol: 'usdt',
    name: 'Tether',
    image: 'https://assets.coingecko.com/coins/images/325/large/Tether.png',
    current_price: 1.00,
    market_cap: 93541876452,
    market_cap_rank: 3,
    price_change_percentage_24h: 0.02,
    price_change_percentage_7d_in_currency: 0.05,
    price_change_percentage_30d_in_currency: -0.03,
    price_change_percentage_1y_in_currency: 0.12,
    total_volume: 65432187651,
    circulating_supply: 93543876000,
  },
];

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

// Function to handle API errors consistently
const handleApiError = (error: any, fallbackData: any = null, endpoint: string): any => {
  // Check if we have a specific error we can handle
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;
    if (axiosError.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error(`API Error (${endpoint}): ${axiosError.response.status} - ${JSON.stringify(axiosError.response.data)}`);
      
      if (axiosError.response.status === 429) {
        console.warn('Rate limit exceeded. Consider upgrading to a paid API plan.');
      }
    } else if (axiosError.request) {
      // The request was made but no response was received
      console.error(`Network Error (${endpoint}): No response received`);
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error(`Request Error (${endpoint}): ${axiosError.message}`);
    }
  } else {
    // Handle non-Axios errors
    console.error(`Unexpected error (${endpoint}):`, error);
  }
  
  // Return fallback data if provided
  if (fallbackData !== null) {
    console.info(`Using fallback data for ${endpoint}`);
    return fallbackData;
  }
  
  // Re-throw the error if no fallback data
  throw error;
};

// API functions
export const coinGeckoAPI = {
  // Ping the API to check if it's working
  ping: async () => {
    try {
      const response = await api.get('/ping');
      return response.data;
    } catch (error) {
      return handleApiError(error, { status: 'error', gecko_says: '(:./)' }, 'ping');
    }
  },

  // Get list of supported coins
  getCoins: async () => {
    try {
      const response = await api.get('/coins/list');
      return response.data;
    } catch (error) {
      return handleApiError(error, [], 'coins/list');
    }
  },

  // Get market data for coins (with pagination)
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
      
      // Validate the response data structure
      if (!Array.isArray(response.data)) {
        console.error('API returned unexpected data structure', response.data);
        return fallbackCoins;
      }
      
      return response.data as Coin[];
    } catch (error) {
      console.error('Error fetching coin markets:', error);
      // Return fallback data instead of crashing
      return fallbackCoins;
    }
  },

  // Get detailed data for a specific coin
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
      // For individual coin details, just report the error rather than using fallback
      return handleApiError(error, null, `coins/${coinId}`);
    }
  },

  // Get price data for multiple coins
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
      return handleApiError(error, {}, 'simple/price');
    }
  },

  // Get historical chart data for a specific coin
  getCoinChart: async (coinId: string, days = 7, currency = 'usd') => {
    try {
      const response = await api.get(`/coins/${coinId}/market_chart`, {
        params: {
          vs_currency: currency,
          days: days,
        },
      });
      
      // Validate chart data structure
      if (!response.data || !Array.isArray(response.data.prices)) {
        throw new Error('Invalid chart data structure');
      }
      
      return response.data as ChartData;
    } catch (error) {
      // Generate synthetic chart data as fallback
      console.error(`Error fetching chart data for coin ${coinId}:`, error);
      return generateFallbackChartData(days);
    }
  },

  // Additional method for date range chart data
  getCoinChartRange: async (
    coinId: string, 
    from: number, // Unix timestamp in seconds
    to: number, // Unix timestamp in seconds
    currency = 'usd'
  ) => {
    try {
      const response = await api.get(`/coins/${coinId}/market_chart/range`, {
        params: {
          vs_currency: currency,
          from,
          to,
        },
      });
      
      // Validate chart data structure
      if (!response.data || !Array.isArray(response.data.prices)) {
        throw new Error('Invalid chart data structure');
      }
      
      return response.data as ChartData;
    } catch (error) {
      // Generate synthetic chart data as fallback
      console.error(`Error fetching chart range data for coin ${coinId}:`, error);
      return generateFallbackChartData(Math.floor((to - from) / (60 * 60 * 24)));
    }
  },

  // Search for coins, categories, and exchanges
  searchCoins: async (query: string) => {
    try {
      const response = await api.get('/search', {
        params: {
          query,
        },
      });
      return response.data;
    } catch (error) {
      return handleApiError(error, { coins: [], categories: [], exchanges: [] }, 'search');
    }
  },

  // Get trending coins
  getTrendingCoins: async () => {
    try {
      const response = await api.get('/search/trending');
      return response.data;
    } catch (error) {
      return handleApiError(error, { coins: [] }, 'search/trending');
    }
  },
};

// Generate fallback chart data when API fails
function generateFallbackChartData(days: number): ChartData {
  const now = Date.now();
  const prices: [number, number][] = [];
  const market_caps: [number, number][] = [];
  const total_volumes: [number, number][] = [];
  
  // Generate realistic looking price data
  let price = 1000 + Math.random() * 1000;
  const volatility = 0.02;
  
  for (let i = 0; i < days; i++) {
    // Add some random movement to price
    price = price * (1 + (Math.random() * volatility * 2 - volatility));
    
    // Timestamp for this data point (going backward from now)
    const timestamp = now - ((days - i) * 24 * 60 * 60 * 1000);
    
    // Add data point
    prices.push([timestamp, price]);
    market_caps.push([timestamp, price * 1000000]);
    total_volumes.push([timestamp, price * 100000]);
  }
  
  return {
    prices,
    market_caps,
    total_volumes,
  };
}