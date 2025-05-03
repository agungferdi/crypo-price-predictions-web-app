import React from 'react';
import { Coin } from '../services/api';
import CoinCard from './CoinCard';

interface CoinListProps {
  coins: Coin[];
  isLoading: boolean;
  error: Error | null;
  currency: string;
}

const CoinList: React.FC<CoinListProps> = ({ coins, isLoading, error, currency }) => {
  if (isLoading) {
    return <div className="loading-state">Loading coins data...</div>;
  }

  if (error) {
    return (
      <div className="error-state">
        <p>Error loading coin data:</p>
        <p>{error.message}</p>
        <p>Please try again later.</p>
      </div>
    );
  }

  if (coins.length === 0) {
    return <div className="empty-state">No coins found.</div>;
  }

  return (
    <div className="coin-list">
      {coins.map(coin => (
        <CoinCard key={coin.id} coin={coin} currency={currency} />
      ))}
    </div>
  );
};

export default CoinList;