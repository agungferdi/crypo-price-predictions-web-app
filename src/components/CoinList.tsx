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
        <div className="loading-indicator"></div>
        <p>Loading cryptocurrency data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-state">
        <p>Error loading cryptocurrency data:</p>
        <p>{error.message}</p>
        <p>Please try again later or check your connection.</p>
      </div>
    );
  }

  if (!coins || coins.length === 0) {
    return (
      <div className="empty-state">
        <p>No cryptocurrencies found.</p>
        <p>Try changing your search terms or try again later.</p>
      </div>
    );
  }

  return (
    <div className="coin-list">
      {coins.map(coin => (
        // Wrap each card in its own error boundary to prevent one card from crashing all
        <ErrorBoundary key={`error-boundary-${coin.id}`} fallback={
          <div className="coin-card-error">
            <p>Failed to display {coin.name}</p>
            <p>Please reload the page to try again.</p>
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