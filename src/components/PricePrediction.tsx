import React, { useState, useEffect } from 'react';
import { ChartData } from '../services/api';
import { predictionsAPI, ModelMetrics, PredictionResult } from '../services/predictionsApi';
import * as tf from '@tensorflow/tfjs';

export type TimeFrame = '1d' | '7d' | '30d' | '365d' | '2y' | '4y';

interface PricePredictionProps {
  coinId: string;
  historicalData: ChartData | null;
  currency: string;
}

const PricePrediction: React.FC<PricePredictionProps> = ({ coinId, historicalData, currency }) => {
  const [predictions, setPredictions] = useState<Record<TimeFrame, number | null>>({
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

  useEffect(() => {
    if (coinId) {
      checkTensorflowSupport();
    }
  }, []);

  useEffect(() => {
    if (coinId) {
      fetchPredictions();
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

  const fetchPredictions = async () => {
    if (!coinId) return;
    
    setIsLoading(true);
    setError(null);
    setModelStatus('Initializing...');
    
    try {
      const isTfAvailable = await predictionsAPI.isApiAvailable();
      
      if (!isTfAvailable) {
        setError('Browser does not support TensorFlow.js');
        setIsLoading(false);
        return;
      }
      
      setModelStatus('Training model...');
      
      const result = await predictionsAPI.getPredictions(coinId, currency);
      
      setPredictions(result.predictions);
      setConfidence(result.confidence);
      setMetrics(result.metrics);
      setChangePercentages(result.changePercentages);
      setModelStatus('AI model ready!');
      
    } catch (err) {
      console.error('Error fetching predictions:', err);
      setError('Failed to generate predictions');
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
    <div className="prediction-container">
      <div className="prediction-header">
        <div className="prediction-icon">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 2v20h20"></path>
            <path d="M5 16l3-3 3 3 8-8"></path>
            <path d="M15 8h4v4"></path>
          </svg>
        </div>
        <h3 className="prediction-title">AI Price Forecast</h3>
        <span className="model-badge">{modelStatus}</span>
      </div>
      
      {isLoading ? (
        <div className="training-indicator">
          <div className="ai-pulse-animation"></div>
          <div className="training-progress">
            <div className="progress-bar"></div>
          </div>
          <p>AI model in training</p>
        </div>
      ) : error ? (
        <div className="prediction-error">
          <div className="error-icon-small">⚠️</div>
          <p>{error}</p>
          <button 
            className="retry-button-small"
            onClick={fetchPredictions}
          >
            Retry
          </button>
        </div>
      ) : predictions['1d'] !== null ? (
        <div className="prediction-result">
          <div className="prediction-timeframes">
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
          
          <div className="prediction-price-display">
            <div className="prediction-label">
              {selectedTimeFrame === '1d' ? 'Next 24h:' : 
               selectedTimeFrame === '7d' ? 'Next Week:' :
               selectedTimeFrame === '30d' ? 'Next Month:' : 'Next Year:'}
            </div>
            <div className="prediction-price">
              {formatCurrency(predictions[selectedTimeFrame])}
            </div>
            <div className={`prediction-change-badge ${getChangeClass(selectedTimeFrame)}`}>
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
          
          <div className="prediction-footer">
            <div className="prediction-info-toggle" onClick={() => document.getElementById(`metrics-${coinId}`)?.classList.toggle('visible')}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 8v4M12 16h.01"></path>
              </svg>
              <span>AI model info</span>
            </div>
            
            <div id={`metrics-${coinId}`} className="metrics-popup">
              <div className="metrics-header">
                <h4>AI Model Details</h4>
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
                  Predictions based on historical patterns only. Not financial advice.
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="prediction-loading">
          <button className="get-predictions-button" onClick={fetchPredictions}>
            Generate AI Prediction
          </button>
        </div>
      )}
    </div>
  );
};

export default PricePrediction;