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
  const [tfSupported, setTfSupported] = useState<boolean | null>(null);

  // Only check TensorFlow support once on initial mount
  useEffect(() => {
    const checkTensorflowSupport = async () => {
      try {
        // Use a simple try/catch to check if TF is available
        if (tf && typeof tf.ready === 'function') {
          await tf.ready();
          setTfSupported(true);
          setModelStatus('TensorFlow.js is ready');
        } else {
          setTfSupported(false);
          setError('TensorFlow.js is not supported in this browser');
        }
      } catch (err) {
        setTfSupported(false);
        setError('TensorFlow.js is not supported in this browser');
        console.error('TensorFlow.js not supported:', err);
      }
    };
    
    checkTensorflowSupport();
  }, []);

  // Only fetch predictions when coinId or currency changes and TF is supported
  useEffect(() => {
    if (coinId && tfSupported === true) {
      fetchPredictions();
    }
  }, [coinId, currency, tfSupported]);

  const fetchPredictions = async () => {
    if (!coinId || tfSupported === false) return;
    
    setIsLoading(true);
    setError(null);
    setModelStatus('Preparing optimized model...');
    
    try {
      // Check if API is available first
      const isTfAvailable = await predictionsAPI.isApiAvailable().catch(() => false);
      
      if (!isTfAvailable) {
        setError('Prediction API is not available. Try again later.');
        setIsLoading(false);
        return;
      }
      
      setModelStatus('Training lightweight model with recent data...');
      
      // Get predictions with a timeout to prevent long-running operations
      const fetchWithTimeout = new Promise<PredictionResult>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Prediction timed out'));
        }, 15000); // 15 second timeout
        
        predictionsAPI.getPredictions(coinId, currency)
          .then(result => {
            clearTimeout(timeout);
            resolve(result);
          })
          .catch(err => {
            clearTimeout(timeout);
            reject(err);
          });
      });
      
      const result = await fetchWithTimeout;
      
      // Update state with predictions
      setPredictions(result.predictions);
      setConfidence(result.confidence);
      setMetrics(result.metrics);
      setChangePercentages(result.changePercentages);
      setModelStatus('Prediction complete! Using optimized ML model.');
      
    } catch (err) {
      console.error('Error fetching predictions:', err);
      setError('Failed to generate price predictions');
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

  return (
    <div className="prediction-container">
      <h3 className="prediction-title">AI Price Predictions</h3>
      <p className="model-status">{modelStatus}</p>
      
      {isLoading ? (
        <div className="training-indicator">
          <div className="spinner"></div>
          <p>Training model and generating predictions...</p>
          <p className="training-note">This may take a few moments as calculations are performed in your browser</p>
        </div>
      ) : error ? (
        <div className="prediction-error">
          {error}
          <button 
            className="retry-button"
            onClick={fetchPredictions}
            disabled={tfSupported === false}
          >
            Retry
          </button>
        </div>
      ) : predictions['1d'] !== null ? (
        <div className="prediction-result">
          <div className="timeframe-selector">
            <button 
              className={selectedTimeFrame === '1d' ? 'active' : ''}
              onClick={() => setSelectedTimeFrame('1d')}
            >
              24h
            </button>
            <button 
              className={selectedTimeFrame === '7d' ? 'active' : ''}
              onClick={() => setSelectedTimeFrame('7d')}
            >
              7d
            </button>
            <button 
              className={selectedTimeFrame === '30d' ? 'active' : ''}
              onClick={() => setSelectedTimeFrame('30d')}
            >
              30d
            </button>
            <button 
              className={selectedTimeFrame === '365d' ? 'active' : ''}
              onClick={() => setSelectedTimeFrame('365d')}
            >
              1y
            </button>
            <button 
              className={selectedTimeFrame === '2y' ? 'active' : ''}
              onClick={() => setSelectedTimeFrame('2y')}
            >
              2y
            </button>
            <button 
              className={selectedTimeFrame === '4y' ? 'active' : ''}
              onClick={() => setSelectedTimeFrame('4y')}
            >
              4y
            </button>
          </div>
          
          <div className="prediction-value">
            <span className="prediction-label">
              {selectedTimeFrame === '1d' ? 'Next 24h:' : 
               selectedTimeFrame === '7d' ? 'Next 7 days:' :
               selectedTimeFrame === '30d' ? 'Next 30 days:' :
               selectedTimeFrame === '365d' ? 'Next year:' :
               selectedTimeFrame === '2y' ? 'Next 2 years:' : 'Next 4 years:'}
            </span>
            <span className="prediction-price">
              {formatCurrency(predictions[selectedTimeFrame])}
              <span className={`prediction-change ${getChangeClass(selectedTimeFrame)}`}>
                {changePercentages[selectedTimeFrame]}
              </span>
            </span>
          </div>
          
          <div className="confidence-meter">
            <div className="confidence-bar" style={{ width: `${confidence}%` }}></div>
            <div className="confidence-label">{confidence.toFixed(0)}% confidence</div>
          </div>
          
          <div className="prediction-disclaimer">
            Prediction based on historical patterns only. Not financial advice.
            <p className="browser-note">All machine learning is performed directly in your browser with TensorFlow.js</p>
          </div>
          
          {metrics && (
            <div className="model-metrics">
              <details>
                <summary>Model Details</summary>
                <div className="metrics-content">
                  <p>Mean Square Error: {metrics.mse.toFixed(4)}</p>
                  <p>Mean Absolute Error: {metrics.mae.toFixed(4)}</p>
                  <p>Root Mean Square Error: {metrics.rmse.toFixed(4)}</p>
                </div>
              </details>
            </div>
          )}
        </div>
      ) : (
        <div className="prediction-loading">
          <button 
            className="get-predictions-button" 
            onClick={fetchPredictions}
            disabled={tfSupported === false}
          >
            Train Model & Get Predictions
          </button>
        </div>
      )}
    </div>
  );
};

export default PricePrediction;