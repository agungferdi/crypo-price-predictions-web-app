import React from 'react';
import { Coin } from '../services/api';
import CoinCard from './CoinCard';
import ErrorBoundary from './ErrorBoundary/index';

interface CoinListProps {
  coins: Coin[];
  isLoading: boolean;
  error: Error | null;
  currency: string;
}

const CoinList: React.FC<CoinListProps> = ({ coins, isLoading, error, currency }) => {
  if (isLoading) {
    return (
      <div className="loading-state">
        <div className="loading-spinner"></div>
        <h3>Loading Cryptocurrency Data</h3>
        <p>Fetching the latest market information...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-state">
        <div className="error-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h3>Unable to Load Data</h3>
        <p className="error-message">{error.message}</p>
        <p>Please check your connection and try again.</p>
        <button className="retry-button" onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    );
  }

  if (!coins || coins.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
        </div>
        <h3>No Cryptocurrencies Found</h3>
        <p>Try changing your search terms or check back later.</p>
      </div>
    );
  }

  return (
    <div className="coin-list">
      {coins.map(coin => (
        // Wrap each card in its own error boundary to prevent one card from crashing all
        <ErrorBoundary key={`error-boundary-${coin.id}`} fallback={
          <div className="coin-card-error">
            <h3>Display Error</h3>
            <p>Unable to display {coin.name} data</p>
            <button onClick={() => window.location.reload()} className="retry-button-small">
              Reload
            </button>
          </div>
        }>
          {/* Use React's key prop with a stable unique ID */}
          <CoinCard 
            key={`coin-${coin.id}`}
            coin={coin} 
            currency={currency} 
          />
        </ErrorBoundary>
      ))}
    </div>
  );
};

// Use React.memo to prevent unnecessary re-renders of the entire list
export default React.memo(CoinList);