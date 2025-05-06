import React, { useState, useEffect, useRef, memo } from 'react';
import { Coin, coinGeckoAPI, ChartData } from '../services/api';
import CoinChart from './CoinChart';
import PriceForecast from './PriceForecast';

interface CoinCardProps {
  coin: Coin;
  currency: string;
}

// Using memo to prevent unnecessary rerenders across all cards
const CoinCard: React.FC<CoinCardProps> = memo(({ coin, currency }) => {
  // Create a unique ID for this component instance
  const componentId = useRef(`coin-${coin.id}-${Math.random().toString(36).substring(2, 9)}`);
  
  const [showChart, setShowChart] = useState(false);
  const [showForecast, setShowForecast] = useState(false);
  const [historicalData, setHistoricalData] = useState<ChartData | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [hasAnimated, setHasAnimated] = useState(false);

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

  // Format currency value with proper symbol
  const formatCurrency = (value: number) => {
    const formatter = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency.toUpperCase(),
      notation: value > 1000000 ? 'compact' : 'standard',
      maximumFractionDigits: value < 1 ? 6 : 2,
    });
    
    // Handle BTC and other crypto currencies that don't have standard formats
    if (currency === 'btc') {
      return `₿ ${value.toLocaleString(undefined, { maximumFractionDigits: 8 })}`;
    }
    
    return formatter.format(value);
  };

  // Toggle chart visibility for this card only
  const toggleChart = () => {
    // Close forecast if it's open
    if (showForecast) setShowForecast(false);
    // Toggle chart state
    setShowChart(prev => !prev);
    if (!hasAnimated) setHasAnimated(true);
  };

  // Toggle forecast visibility for this card only
  const toggleForecast = () => {
    // Close chart if it's open
    if (showChart) setShowChart(false);
    // Toggle forecast state
    setShowForecast(prev => !prev);
    if (!hasAnimated) setHasAnimated(true);
  };

  // Fetch historical data for forecasts and chart when needed
  useEffect(() => {
    if ((showChart || showForecast) && !historicalData && !isLoadingData) {
      fetchHistoricalData();
    }
  }, [showChart, showForecast]);

  const fetchHistoricalData = async () => {
    setIsLoadingData(true);
    try {
      // Fetch 30 days of historical data for better forecast
      const data = await coinGeckoAPI.getCoinChart(coin.id, 30, currency);
      setHistoricalData(data);
    } catch (error) {
      console.error(`Error fetching historical data for ${coin.id}:`, error);
    } finally {
      setIsLoadingData(false);
    }
  };
  
  return (
    <div className={`coin-card ${hasAnimated ? 'has-expanded' : ''}`} data-coin-id={coin.id} id={componentId.current}>
      <div className="coin-rank">{coin.market_cap_rank}</div>
      <div className="coin-card-header">
        <div className="coin-image">
          <img src={coin.image} alt={coin.name} loading="lazy" />
        </div>
        <div className="coin-info">
          <h2>{coin.name} <span className="coin-symbol">{coin.symbol.toUpperCase()}</span></h2>
          <div className="coin-price">{formatCurrency(coin.current_price)}</div>
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
          Market Cap: {formatCurrency(coin.market_cap)}
        </div>
        <div className="card-buttons">
          <button 
            className={`chart-toggle-button ${showChart ? 'active' : ''}`}
            onClick={toggleChart}
            aria-pressed={showChart}
            aria-label={`${showChart ? 'Hide' : 'Show'} price chart for ${coin.name}`}
          >
            {showChart ? 'Hide Chart' : 'Chart'}
          </button>
          <button 
            className={`prediction-toggle-button ${showForecast ? 'active' : ''}`}
            onClick={toggleForecast}
            aria-pressed={showForecast}
            aria-label={`${showForecast ? 'Hide' : 'Show'} price forecast for ${coin.name}`}
          >
            {showForecast ? 'Hide Forecast' : 'ML Forecast'}
          </button>
        </div>
      </div>
      
      {showChart && (
        <div className="coin-chart">
          <CoinChart coinId={coin.id} currency={currency} />
        </div>
      )}

      {showForecast && (
        <div className="coin-forecast">
          <PriceForecast 
            coinId={coin.id} 
            historicalData={historicalData}
            currency={currency} 
          />
        </div>
      )}
    </div>
  );
});

export default CoinCard;