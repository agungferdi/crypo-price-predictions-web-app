import React, { useState, useEffect, useRef } from 'react';
import { ChartData } from '../services/api';
import { forecastsAPI, ForecastResult } from '../services/forecastsApi';
import './PriceForecast.css';

// Timeframes for forecasts
export type TimeFrame = '1d' | '7d' | '30d' | '365d' | '2y' | '4y';

interface PriceForecastProps {
  coinId: string;
  currency: string;
  historicalData?: ChartData | null;
}

const PriceForecast: React.FC<PriceForecastProps> = ({ coinId, currency, historicalData }) => {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeFrame>('1d');
  const [forecastResult, setForecastResult] = useState<ForecastResult | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<{
    step: string;
    progress: number;
    total: number;
    logs?: any;
  }>({ step: 'initializing', progress: 0, total: 100 });
  const [showModelInfo, setShowModelInfo] = useState<boolean>(false);
  const timeframesRef = useRef<HTMLDivElement>(null);

  // Function to format currency values
  const formatCurrency = (value: number) => {
    if (!value && value !== 0) return 'N/A';
    
    const formatter = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency.toUpperCase(),
      minimumFractionDigits: value < 1 ? 6 : 2,
      maximumFractionDigits: value < 1 ? 6 : 2,
    });
    return formatter.format(value);
  };

  // Get forecasts on component mount
  useEffect(() => {
    const fetchForecasts = async () => {
      try {
        setIsLoading(true);
        
        // Check if TensorFlow.js is available in the browser
        const apiAvailable = await forecastsAPI.isApiAvailable();
        if (!apiAvailable) {
          console.error('TensorFlow.js not available in this browser');
          setIsLoading(false);
          return;
        }

        // Get forecasts with progress tracking
        const result = await forecastsAPI.getPredictions(
          coinId,
          currency,
          180,
          (step, epoch, total, logs) => {
            setTrainingProgress({
              step,
              progress: epoch,
              total,
              logs,
            });
          }
        );

        setForecastResult(result);
        setIsLoading(false);
      } catch (error) {
        console.error('Error getting forecasts:', error);
        setIsLoading(false);
      }
    };

    fetchForecasts();
  }, [coinId, currency]);

  // Handle timeframe selection
  const handleTimeframeChange = (timeframe: TimeFrame) => {
    setSelectedTimeframe(timeframe);
  };

  // Toggle model info visibility
  const toggleModelInfo = () => {
    setShowModelInfo(prev => !prev);
  };

  // Close model info when clicking outside
  const handleClickOutside = (event: MouseEvent) => {
    const metricsPopup = document.getElementById(`metrics-${coinId}`);
    if (showModelInfo && metricsPopup && !metricsPopup.contains(event.target as Node)) {
      setShowModelInfo(false);
    }
  };

  // Add event listener for clicking outside
  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showModelInfo, coinId]);

  // Main render function
  const renderForecastContent = () => {
    if (isLoading) {
      return renderTrainingStatus();
    }

    if (!forecastResult) {
      return <div className="training-indicator">Failed to generate forecast</div>;
    }

    return (
      <div className="forecast-container">
        <div className="forecast-header">
          <div className="forecast-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 2v20h20"></path>
              <path d="M5 16l3-3 3 3 8-8"></path>
              <path d="M15 8h4v4"></path>
            </svg>
          </div>
          <h3 className="forecast-title">ML Price Forecast</h3>
          <span className="model-badge">ML model ready!</span>
        </div>
        
        <div className="forecast-result">
          <div className="forecast-timeframes">
            <button 
              className={`timeframe-pill ${selectedTimeframe === '1d' ? 'active' : ''}`}
              onClick={() => handleTimeframeChange('1d')}
            >
              24h
            </button>
            <button 
              className={`timeframe-pill ${selectedTimeframe === '7d' ? 'active' : ''}`}
              onClick={() => handleTimeframeChange('7d')}
            >
              1W
            </button>
            <button 
              className={`timeframe-pill ${selectedTimeframe === '30d' ? 'active' : ''}`}
              onClick={() => handleTimeframeChange('30d')}
            >
              1M
            </button>
            <button 
              className={`timeframe-pill ${selectedTimeframe === '365d' ? 'active' : ''}`}
              onClick={() => handleTimeframeChange('365d')}
            >
              1Y
            </button>
          </div>
          
          <div className="forecast-price-display">
            <div className="forecast-label">
              {selectedTimeframe === '1d' ? 'Next 24h:' : 
               selectedTimeframe === '7d' ? 'Next Week:' : 
               selectedTimeframe === '30d' ? 'Next Month:' : 
               'Next Year:'}
            </div>
            <div className="forecast-price">
              {formatCurrency(forecastResult.predictions[selectedTimeframe])}
            </div>
            <div className={`forecast-change-badge ${forecastResult.changePercentages[selectedTimeframe].startsWith('+') ? 'positive' : 'negative'}`}>
              <span className="change-icon">{forecastResult.changePercentages[selectedTimeframe].startsWith('+') ? '↗' : '↘'}</span>
              {forecastResult.changePercentages[selectedTimeframe]}
            </div>
          </div>
          
          <div className="confidence-wrapper">
            <div className="confidence-meter">
              <div className="confidence-bar" style={{ width: `${forecastResult.confidence}%` }}></div>
            </div>
            <div className="confidence-value">{Math.round(forecastResult.confidence)}% confidence</div>
          </div>
          
          <div className="forecast-footer">
            <div className="forecast-info-toggle" onClick={toggleModelInfo}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 8v4M12 16h.01"></path>
              </svg>
              <span>ML model info</span>
            </div>
            <div id={`metrics-${coinId}`} className={`metrics-popup ${showModelInfo ? 'visible' : ''}`}>
              <div className="metrics-header">
                <h4>ML Model Details</h4>
                <div className="metrics-close" onClick={() => setShowModelInfo(false)}>✕</div>
              </div>
              <div className="metrics-content">
                <div className="metric-item">
                  <span>Model Type:</span>
                  <span>Regression ML</span>
                </div>
                <div className="metric-item">
                  <span>Mean Abs. Error:</span>
                  <span>{forecastResult.metrics.mae.toFixed(4)}</span>
                </div>
                <div className="metric-item">
                  <span>Root Mean Sq. Error:</span>
                  <span>{forecastResult.metrics.rmse.toFixed(4)}</span>
                </div>
                <div className="metrics-note">
                  Forecasts based on historical patterns only. Not financial advice.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Render training progress
  const renderTrainingStatus = () => {
    let progressPercent = 0;
    
    switch (trainingProgress.step) {
      case 'loading':
        progressPercent = (trainingProgress.progress / trainingProgress.total) * 25;
        break;
      case 'preparing':
        progressPercent = 25 + (trainingProgress.progress / trainingProgress.total) * 15;
        break;
      case 'training':
        progressPercent = 40 + (trainingProgress.progress / trainingProgress.total) * 50;
        break;
      case 'predicting':
        progressPercent = 90 + (trainingProgress.progress / trainingProgress.total) * 10;
        break;
      default:
        progressPercent = 5;
    }
    
    return (
      <div className="forecast-container">
        <div className="forecast-header">
          <div className="forecast-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 2v20h20"></path>
              <path d="M5 16l3-3 3 3 8-8"></path>
              <path d="M15 8h4v4"></path>
            </svg>
          </div>
          <h3 className="forecast-title">ML Price Forecast</h3>
          <span className="model-badge">Training model...</span>
        </div>
        
        <div className="training-indicator">
          <div className="ai-pulse-animation">
            <span className="progress-value">{Math.round(progressPercent)}%</span>
          </div>
          <div className="training-progress">
            <div className="progress-bar" style={{ width: `${progressPercent}%` }}></div>
          </div>
          <div className="training-step">
            {trainingProgress.step === 'loading' && 'Loading historical data...'}
            {trainingProgress.step === 'preparing' && 'Preparing training data...'}
            {trainingProgress.step === 'training' && 'Training ML model...'}
            {trainingProgress.step === 'predicting' && 'Generating forecasts...'}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="coin-forecast">
      {renderForecastContent()}
    </div>
  );
};

export default PriceForecast;