import React, { useState, useEffect } from 'react';
import { ChartData } from '../services/api';
import { predictionsAPI, ModelMetrics } from '../services/predictionsApi';
import * as tf from '@tensorflow/tfjs';

export type TimeFrame = '1d' | '7d' | '30d' | '365d' | '2y' | '4y';

interface PriceForecastProps {
  coinId: string;
  currency: string;
}

const PriceForecast: React.FC<PriceForecastProps> = ({ coinId, currency }) => {
  const [forecasts, setForecasts] = useState<Record<TimeFrame, number | null>>({
    '1d': null,
    '7d': null,
    '30d': null,
    '365d': null,
    '2y': null,
    '4y': null
  });
  const [isLoading, setIsLoading] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<TimeFrame>('1d');
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [changePercentages, setChangePercentages] = useState<Record<TimeFrame, string>>({} as Record<TimeFrame, string>);
  const [modelStatus, setModelStatus] = useState<string>('');
  
  // New states for tracking training progress
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [processingStep, setProcessingStep] = useState('Initializing...');

  useEffect(() => {
    if (coinId) {
      checkTensorflowSupport();
    }
  }, []);

  useEffect(() => {
    if (coinId) {
      fetchForecasts();
    }
  }, [coinId, currency]);

  const checkTensorflowSupport = async () => {
    try {
      await tf.ready();
      setModelStatus('TF.js ready');
    } catch (err) {
      setError('TensorFlow.js is not supported in this browser');
      console.error('TensorFlow.js not supported:', err);
    }
  };

  const fetchForecasts = async () => {
    if (!coinId) return;
    
    setIsLoading(true);
    setError(null);
    setModelStatus('Initializing...');
    setTrainingProgress(0);
    setCurrentEpoch(0);
    setProcessingStep('Loading data');
    
    try {
      const isTfAvailable = await predictionsAPI.isApiAvailable();
      
      if (!isTfAvailable) {
        setError('Browser does not support TensorFlow.js');
        setIsLoading(false);
        return;
      }
      
      setModelStatus('Training model...');
      setProcessingStep('Preparing model');
      
      // Setup progress callback function
      const progressCallback = (step: string, epoch: number, totalEpochs: number, logs?: any) => {
        // Calculate overall progress (data loading: 10%, model setup: 10%, training: 60%, forecast: 20%)
        let progress = 0;
        
        if (step === 'loading') {
          progress = 10 * (epoch / totalEpochs); // 0-10%
          setProcessingStep('Loading historical data');
        } else if (step === 'preparing') {
          progress = 10 + 10 * (epoch / totalEpochs); // 10-20%
          setProcessingStep('Preparing training data');
        } else if (step === 'training') {
          progress = 20 + 60 * (epoch / totalEpochs); // 20-80%
          setCurrentEpoch(epoch);
          setTotalEpochs(totalEpochs);
          setProcessingStep(`Training epoch ${epoch}/${totalEpochs}`);
        } else if (step === 'predicting') {
          progress = 80 + 20 * (epoch / totalEpochs); // 80-100%
          setProcessingStep('Generating forecasts');
        }
        
        setTrainingProgress(Math.min(Math.round(progress), 99)); // Cap at 99% until complete
      };
      
      const result = await predictionsAPI.getPredictions(coinId, currency, 180, progressCallback);
      
      // Set final progress to 100%
      setTrainingProgress(100);
      setProcessingStep('Complete');
      
      setForecasts(result.predictions);
      setConfidence(result.confidence);
      setMetrics(result.metrics);
      setChangePercentages(result.changePercentages);
      setModelStatus('ML model ready!');
      
    } catch (err) {
      console.error('Error fetching forecasts:', err);
      setError('Failed to generate forecasts');
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrency = (value: number | null): string => {
    if (value === null) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency.toUpperCase(),
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    }).format(value);
  };

  const getChangeClass = (timeFrame: TimeFrame): string => {
    if (!changePercentages[timeFrame]) return '';
    return changePercentages[timeFrame].startsWith('+') ? 'positive' : 'negative';
  };

  const getIconForChange = (timeFrame: TimeFrame): string => {
    if (!changePercentages[timeFrame]) return '';
    return changePercentages[timeFrame].startsWith('+') 
      ? '↗' 
      : '↘';
  };

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
        <span className="model-badge">{modelStatus}</span>
      </div>
      
      {isLoading ? (
        <div className="training-indicator">
          <div 
            className="ai-pulse-animation"
            style={{
              opacity: 0.3 + (trainingProgress / 100) * 0.7,
              transform: `scale(${0.85 + (trainingProgress / 100) * 0.15})`
            }}
          >
            <span className="progress-value">{trainingProgress}%</span>
          </div>
          <div className="training-progress">
            <div 
              className="progress-bar"
              style={{ width: `${trainingProgress}%` }}
            ></div>
          </div>
          <p>{processingStep}</p>
          {currentEpoch > 0 && totalEpochs > 0 && (
            <p className="epoch-display">
              <small>Epoch: {currentEpoch}/{totalEpochs}</small>
            </p>
          )}
        </div>
      ) : error ? (
        <div className="forecast-error">
          <div className="error-icon-small">⚠️</div>
          <p>{error}</p>
          <button 
            className="retry-button-small"
            onClick={fetchForecasts}
          >
            Retry
          </button>
        </div>
      ) : forecasts['1d'] !== null ? (
        <div className="forecast-result">
          <div className="forecast-timeframes">
            {['1d', '7d', '30d', '365d'].map((tf) => (
              <button 
                key={tf}
                className={`timeframe-pill ${selectedTimeFrame === tf ? 'active' : ''}`}
                onClick={() => setSelectedTimeFrame(tf as TimeFrame)}
              >
                {tf === '1d' ? '24h' : 
                 tf === '7d' ? '1W' :
                 tf === '30d' ? '1M' : '1Y'}
              </button>
            ))}
          </div>
          
          <div className="forecast-price-display">
            <div className="forecast-label">
              {selectedTimeFrame === '1d' ? 'Next 24h:' : 
               selectedTimeFrame === '7d' ? 'Next Week:' :
               selectedTimeFrame === '30d' ? 'Next Month:' : 'Next Year:'}
            </div>
            <div className="forecast-price">
              {formatCurrency(forecasts[selectedTimeFrame])}
            </div>
            <div className={`forecast-change-badge ${getChangeClass(selectedTimeFrame)}`}>
              <span className="change-icon">{getIconForChange(selectedTimeFrame)}</span>
              {changePercentages[selectedTimeFrame]}
            </div>
          </div>
          
          <div className="confidence-wrapper">
            <div className="confidence-meter">
              <div className="confidence-bar" style={{ width: `${confidence}%` }}></div>
            </div>
            <div className="confidence-value">{confidence.toFixed(0)}% confidence</div>
          </div>
          
          <div className="forecast-footer">
            <div className="forecast-info-toggle" onClick={() => document.getElementById(`metrics-${coinId}`)?.classList.toggle('visible')}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 8v4M12 16h.01"></path>
              </svg>
              <span>ML model info</span>
            </div>
            
            <div id={`metrics-${coinId}`} className="metrics-popup">
              <div className="metrics-header">
                <h4>ML Model Details</h4>
                <div className="metrics-close" onClick={() => document.getElementById(`metrics-${coinId}`)?.classList.remove('visible')}>✕</div>
              </div>
              <div className="metrics-content">
                <div className="metric-item">
                  <span>Model Type:</span>
                  <span>Regression ML</span>
                </div>
                {metrics && (
                  <>
                    <div className="metric-item">
                      <span>Mean Abs. Error:</span>
                      <span>{metrics.mae.toFixed(4)}</span>
                    </div>
                    <div className="metric-item">
                      <span>Root Mean Sq. Error:</span>
                      <span>{metrics.rmse.toFixed(4)}</span>
                    </div>
                  </>
                )}
                <div className="metrics-note">
                  Forecasts based on historical patterns only. Not financial advice.
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="forecast-loading">
          <button className="get-forecasts-button" onClick={fetchForecasts}>
            Generate ML Forecast
          </button>
        </div>
      )}
    </div>
  );
};

export default PriceForecast;