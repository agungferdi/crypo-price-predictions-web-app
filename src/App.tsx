import { useState } from 'react';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { coinGeckoAPI, Coin } from './services/api';
import CoinList from './components/CoinList';
import SearchBar from './components/SearchBar';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
      retry: 3, // Increase retries
      retryDelay: attempt => Math.min(1000 * 2 ** attempt, 30000), // Exponential backoff
    },
  },
});

// Main app wrapper with React Query provider and ErrorBoundary
function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AppContent />
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

// Main app content
function AppContent() {
  const [page, setPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [currency, setCurrency] = useState('usd');
  
  // Fetch coins using React Query with error handling
  const { data: coins, isLoading, error, isError } = useQuery<Coin[]>({
    queryKey: ['coins', page, currency],
    queryFn: async () => {
      try {
        // Add timeout for API calls
        const timeoutPromise = new Promise<Coin[]>((_, reject) => {
          setTimeout(() => reject(new Error('API request timed out')), 20000);
        });
        
        // Race between actual API call and timeout
        return Promise.race([
          coinGeckoAPI.getMarkets(currency, page, 20),
          timeoutPromise
        ]);
      } catch (err) {
        console.error("Error fetching cryptocurrency data:", err);
        throw err;
      }
    },
    keepPreviousData: true,
  });

  // Filter coins based on search term - with safe handling of undefined coins
  const filteredCoins = coins 
    ? coins.filter(coin => 
        coin.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
        coin.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : [];

  // Handle search
  const handleSearch = (query: string) => {
    setSearchTerm(query);
  };

  const handleNextPage = () => {
    setPage(prev => prev + 1);
    window.scrollTo(0, 0);
  };

  const handlePrevPage = () => {
    setPage(prev => Math.max(1, prev - 1));
    window.scrollTo(0, 0);
  };

  return (
    <div className="app-container">
      <header>
        <h1>Crypto Price Tracker</h1>
        <p className="subtitle">Live cryptocurrency prices and market data</p>
      </header>
      
      <div className="controls">
        <SearchBar onSearch={handleSearch} />
        
        <div className="currency-selector">
          <label>Currency:</label>
          <select value={currency} onChange={(e) => setCurrency(e.target.value)}>
            <option value="usd">USD ($)</option>
            <option value="eur">EUR (€)</option>
            <option value="gbp">GBP (£)</option>
            <option value="jpy">JPY (¥)</option>
            <option value="btc">BTC (₿)</option>
          </select>
        </div>
      </div>
      
      <ErrorBoundary>
        <CoinList 
          coins={filteredCoins}
          isLoading={isLoading}
          error={error as Error}
          currency={currency}
        />
      </ErrorBoundary>
      
      {!isLoading && !isError && coins?.length > 0 && (
        <div className="pagination">
          <button onClick={handlePrevPage} disabled={page === 1}>
            Previous Page
          </button>
          <span>Page {page}</span>
          <button onClick={handleNextPage}>
            Next Page
          </button>
        </div>
      )}
      
      <footer>
        <p>Data provided by CoinGecko API</p>
        <p>© 2025 Crypto Price Tracker</p>
      </footer>
    </div>
  );
}

export default App;
