import { useState, useEffect, Suspense } from 'react';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { coinGeckoAPI, Coin } from './services/api';
import CoinList from './components/CoinList';
import SearchBar from './components/SearchBar';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

// Create a client for React Query with better error handling
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
      retry: 3, // Increased retries
      retryDelay: attempt => Math.min(1000 * 2 ** attempt, 30000), // Exponential backoff
      suspense: false, // Disable suspense mode
      useErrorBoundary: false, // Handle errors manually
    },
  },
});

// Fallback component for loading state
const LoadingFallback = () => (
  <div className="loading-fallback">
    <p>Loading application...</p>
  </div>
);

// Error fallback component
const ErrorFallback = ({ error }: { error: Error }) => (
  <div className="error-fallback">
    <h2>Something went wrong</h2>
    <p>{error.message}</p>
    <button onClick={() => window.location.reload()}>Try again</button>
  </div>
);

// Main app wrapper with React Query provider and error handling
function App() {
  return (
    <ErrorBoundary fallback={<ErrorFallback error={new Error("Application crashed")} />}>
      <QueryClientProvider client={queryClient}>
        <Suspense fallback={<LoadingFallback />}>
          <AppContent />
        </Suspense>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

// Main app content
function AppContent() {
  const [page, setPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [currency, setCurrency] = useState('usd');
  const [hasError, setHasError] = useState(false);
  
  // Fetch coins using React Query with better error handling
  const { 
    data: coins, 
    isLoading, 
    error,
    isError
  } = useQuery<Coin[]>({
    queryKey: ['coins', page, currency],
    queryFn: async () => {
      try {
        // Add timeout for API calls
        const timeoutPromise = new Promise<Coin[]>((_, reject) => {
          setTimeout(() => reject(new Error('API request timed out')), 20000);
        });
        
        // Race between actual API call and timeout
        const data = await Promise.race([
          coinGeckoAPI.getMarkets(currency, page, 20),
          timeoutPromise
        ]);
        
        // Ensure we have data
        if (!data || !Array.isArray(data) || data.length === 0) {
          throw new Error('No coin data available');
        }
        
        return data;
      } catch (err) {
        console.error("Error fetching cryptocurrency data:", err);
        setHasError(true);
        throw err;
      }
    },
    keepPreviousData: true,
    onError: (err) => {
      console.error("Query error:", err);
      setHasError(true);
    }
  });

  // Filter coins based on search term with safe handling
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

  // Handle currency change
  const handleCurrencyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setCurrency(e.target.value);
  };

  return (
    <div className="app-container">
      <div className="content-container">
        <header>
          <h1>Crypto Price Tracker</h1>
          <p className="subtitle">
            Track real-time cryptocurrency prices, view historical charts, and get AI-powered price predictions
          </p>
        </header>

        <div className="controls">
          <SearchBar onSearch={handleSearch} />
          <div className="currency-selector">
            <label htmlFor="currency">Currency:</label>
            <select 
              id="currency" 
              value={currency} 
              onChange={handleCurrencyChange}
            >
              <option value="usd">USD</option>
              <option value="eur">EUR</option>
              <option value="jpy">JPY</option>
              <option value="gbp">GBP</option>
              <option value="btc">BTC</option>
            </select>
          </div>
        </div>

        {isLoading ? (
          <div className="loading">
            <p>Loading cryptocurrency data...</p>
          </div>
        ) : isError ? (
          <div className="error-message">
            <p>Error loading data. Please try again later.</p>
            <p className="error-details">{(error as Error)?.message || 'Unknown error'}</p>
            <button 
              className="retry-button"
              onClick={() => window.location.reload()}
            >
              Retry
            </button>
          </div>
        ) : (
          <>
            {filteredCoins.length === 0 && !isLoading ? (
              <div className="no-results">
                <p>No cryptocurrencies found matching your search.</p>
                {searchTerm && (
                  <button 
                    className="clear-search-button"
                    onClick={() => setSearchTerm('')}
                  >
                    Clear Search
                  </button>
                )}
              </div>
            ) : (
              <div className="coin-list-container">
                <CoinList coins={filteredCoins} currency={currency} />
              </div>
            )}
            
            <div className="pagination">
              <button 
                className="pagination-button prev"
                onClick={handlePrevPage}
                disabled={page <= 1}
              >
                Previous
              </button>
              <span className="current-page">Page {page}</span>
              <button 
                className="pagination-button next"
                onClick={handleNextPage}
              >
                Next
              </button>
            </div>
          </>
        )}

        <footer>
          <p>Data provided by CoinGecko API | © {new Date().getFullYear()} Crypto Price Tracker</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
