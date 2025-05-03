import React, { useState, useEffect } from 'react';
import { Coin, coinGeckoAPI, ChartData } from '../services/api';
import CoinChart from './CoinChart';
import PricePrediction from './PricePrediction';

interface CoinCardProps {
  coin: Coin;
  currency: string;
}

const CoinCard: React.FC<CoinCardProps> = ({ coin, currency }) => {
  const [showChart, setShowChart] = useState(false);
  const [showPrediction, setShowPrediction] = useState(false);
  const [historicalData, setHistoricalData] = useState<ChartData | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(false);

  // Format price changes with proper coloring
  const formatPriceChange = (changePercentage: number | undefined | null) => {
    if (changePercentage === undefined || changePercentage === null) {
      return <span className="price-change neutral">N/A</span>;
    }
    
    const formatted = changePercentage.toFixed(2);
    const isPositive = changePercentage >= 0;
    
    return (
      <span className={`price-change ${isPositive ? 'positive' : 'negative'}`}>
        {isPositive ? '+' : ''}{formatted}%
      </span>
    );
  };

  // Fetch historical data for predictions and chart when needed
  useEffect(() => {
    if ((showChart || showPrediction) && !historicalData && !isLoadingData) {
      fetchHistoricalData();
    }
  }, [showChart, showPrediction]);

  const fetchHistoricalData = async () => {
    setIsLoadingData(true);
    try {
      // Fetch 30 days of historical data for better prediction
      const data = await coinGeckoAPI.getCoinChart(coin.id, 30, currency);
      setHistoricalData(data);
    } catch (error) {
      console.error(`Error fetching historical data for ${coin.id}:`, error);
    } finally {
      setIsLoadingData(false);
    }
  };
  
  return (
    <div className="coin-card">
      <div className="coin-rank">{coin.market_cap_rank}</div>
      <div className="coin-card-header">
        <div className="coin-image">
          <img src={coin.image} alt={coin.name} />
        </div>
        <div className="coin-info">
          <h2>{coin.name} <span className="coin-symbol">({coin.symbol.toUpperCase()})</span></h2>
          <div className="coin-price">${coin.current_price.toLocaleString()}</div>
        </div>
      </div>
      
      <div className="coin-price-changes">
        <div className="price-change-row">
          <span className="timeframe">24h:</span>
          {formatPriceChange(coin.price_change_percentage_24h)}
        </div>
        <div className="price-change-row">
          <span className="timeframe">7d:</span>
          {formatPriceChange(coin.price_change_percentage_7d_in_currency)}
        </div>
        <div className="price-change-row">
          <span className="timeframe">30d:</span>
          {formatPriceChange(coin.price_change_percentage_30d_in_currency)}
        </div>
        <div className="price-change-row">
          <span className="timeframe">1y:</span>
          {formatPriceChange(coin.price_change_percentage_1y_in_currency)}
        </div>
      </div>
      
      <div className="coin-card-footer">
        <div className="coin-marketcap">
          Market Cap: ${coin.market_cap.toLocaleString()}
        </div>
        <div className="card-buttons">
          <button 
            className="chart-toggle-button"
            onClick={() => setShowChart(!showChart)}
          >
            {showChart ? 'Hide Chart' : 'Show Chart'}
          </button>
          <button 
            className="prediction-toggle-button"
            onClick={() => setShowPrediction(!showPrediction)}
          >
            {showPrediction ? 'Hide AI Prediction' : 'Show AI Prediction'}
          </button>
        </div>
      </div>
      
      {showChart && (
        <div className="coin-chart">
          <CoinChart coinId={coin.id} currency={currency} />
        </div>
      )}

      {showPrediction && (
        <div className="coin-prediction">
          <PricePrediction 
            coinId={coin.id} 
            historicalData={historicalData}
            currency={currency} 
          />
        </div>
      )}
    </div>
  );
};

export default CoinCard;