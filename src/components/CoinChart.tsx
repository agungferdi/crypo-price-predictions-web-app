import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { coinGeckoAPI, ChartData } from '../services/api';
import './CoinChart.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface CoinChartProps {
  coinId: string;
  currency: string;
}

const CoinChart: React.FC<CoinChartProps> = ({ coinId, currency }) => {
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<number>(7);

  useEffect(() => {
    const fetchChartData = async () => {
      setIsLoading(true);
      try {
        const data = await coinGeckoAPI.getCoinChart(coinId, timeRange, currency);
        setChartData(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching chart data:', err);
        setError('Failed to load chart data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchChartData();
  }, [coinId, currency, timeRange]);

  const formatDate = (timestamp: number): string => {
    const date = new Date(timestamp);
    
    if (timeRange <= 1) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (timeRange <= 7) {
      return date.toLocaleDateString([], { day: 'numeric', month: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            return `${currency.toUpperCase()}: ${context.parsed.y.toFixed(2)}`;
          },
        },
        padding: 10,
        titleFont: {
          size: 12,
          weight: 'bold' as const,
        },
        bodyFont: {
          size: 12,
        },
        backgroundColor: 'rgba(15, 23, 42, 0.8)',
        titleColor: '#e2e8f0',
        bodyColor: '#f1f5f9',
        borderColor: 'rgba(51, 65, 85, 0.5)',
        borderWidth: 1,
        displayColors: false,
      },
    },
    scales: {
      x: {
        ticks: {
          maxTicksLimit: 8,
          color: '#94a3b8',
          font: {
            size: 10,
          },
        },
        grid: {
          display: false,
        },
      },
      y: {
        position: 'right' as const,
        grid: {
          color: 'rgba(51, 65, 85, 0.3)',
        },
        ticks: {
          color: '#94a3b8',
          font: {
            size: 10,
          },
          callback: function(value: any) {
            if (value >= 1000) {
              return '$' + value / 1000 + 'k';
            }
            return '$' + value;
          }
        },
      },
    },
    elements: {
      point: {
        radius: 0,
        hoverRadius: 6,
        hitRadius: 30,
      },
      line: {
        tension: 0.2,
        borderWidth: 2,
      },
    },
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    hover: {
      mode: 'index' as const,
      intersect: false,
    },
  };

  const prepareChartData = () => {
    if (!chartData || !chartData.prices) return null;

    const gradient = document.createElement('canvas').getContext('2d');
    
    if (!gradient) return null;
    
    const gradientFill = gradient.createLinearGradient(0, 0, 0, 350);
    
    const isPositive = chartData.prices[0][1] <= chartData.prices[chartData.prices.length - 1][1];
    
    if (isPositive) {
      gradientFill.addColorStop(0, 'rgba(22, 199, 132, 0.3)');
      gradientFill.addColorStop(1, 'rgba(22, 199, 132, 0)');
    } else {
      gradientFill.addColorStop(0, 'rgba(234, 57, 67, 0.3)');
      gradientFill.addColorStop(1, 'rgba(234, 57, 67, 0)');
    }

    const lineColor = isPositive ? 'rgb(22, 199, 132)' : 'rgb(234, 57, 67)';
    
    return {
      labels: chartData.prices.map(data => formatDate(data[0])),
      datasets: [
        {
          label: `Price (${currency.toUpperCase()})`,
          data: chartData.prices.map(data => data[1]),
          borderColor: lineColor,
          backgroundColor: gradientFill,
          borderWidth: 2,
          fill: true,
          tension: 0.1,
        },
      ],
    };
  };

  const data = prepareChartData();

  return (
    <div className="coin-chart-container">
      <div className="chart-timeframes">
        {[
          { value: 1, label: '24h' },
          { value: 7, label: '7d' },
          { value: 30, label: '1M' },
          { value: 90, label: '3M' },
          { value: 365, label: '1Y' }
        ].map((period) => (
          <button 
            key={period.value}
            className={`timeframe-pill ${timeRange === period.value ? 'active' : ''}`} 
            onClick={() => setTimeRange(period.value)}
          >
            {period.label}
          </button>
        ))}
      </div>

      <div className="chart-wrapper">
        {isLoading && (
          <div className="chart-loading">
            <div className="loading-spinner"></div>
            <p>Loading chart data...</p>
          </div>
        )}
        
        {error && (
          <div className="chart-error">
            <div className="error-icon">⚠️</div>
            <p>{error}</p>
            <button 
              className="retry-button"
              onClick={() => {
                setChartData(null);
                const fetchData = async () => {
                  setIsLoading(true);
                  try {
                    const data = await coinGeckoAPI.getCoinChart(coinId, timeRange, currency);
                    setChartData(data);
                    setError(null);
                  } catch (err) {
                    setError('Failed to load chart data');
                  } finally {
                    setIsLoading(false);
                  }
                };
                fetchData();
              }}
            >
              Retry
            </button>
          </div>
        )}
        
        {!isLoading && !error && data && (
          <Line options={chartOptions} data={data} height={250} />
        )}
      </div>
    </div>
  );
};

export default CoinChart;