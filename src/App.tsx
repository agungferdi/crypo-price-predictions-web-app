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
          <h1>Crypto Price Forecast</h1>
          <p className="subtitle">
            Track real-time cryptocurrency prices, view historical charts, and get Machine Learning powered price forecasts
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

        <footer className="app-footer">
          <p>© {new Date().getFullYear()} Developed by Agung</p>
          <div className="social-links">
            <a href="https://github.com/agungferdi" target="_blank" rel="noopener noreferrer" className="social-link">
              <svg className="social-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              GitHub
            </a>
            <a href="https://www.linkedin.com/in/muhammad-agung-ferdiansyah-/" target="_blank" rel="noopener noreferrer" className="social-link">
              <svg className="social-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
              </svg>
              LinkedIn
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
