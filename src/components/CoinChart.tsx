import React, { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { coinGeckoAPI, ChartData } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface CoinChartProps {
  coinId: string;
  currency: string;
}

const CoinChart: React.FC<CoinChartProps> = ({ coinId, currency }) => {
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<number>(7); // 7 days by default
  const chartRef = useRef<ChartJS>(null);

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
    return date.toLocaleDateString();
  };

  const chartOptions = {
    responsive: true,
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
      },
    },
    scales: {
      x: {
        ticks: {
          maxTicksLimit: 8,
        },
        grid: {
          display: false,
        },
      },
      y: {
        position: 'right' as const,
        grid: {
          color: 'rgba(200, 200, 200, 0.2)',
        },
      },
    },
    elements: {
      point: {
        radius: 0,
      },
    },
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
  };

  const prepareChartData = () => {
    if (!chartData || !chartData.prices) return null;
    
    // Check if prices have data to avoid rendering errors
    if (chartData.prices.length === 0) {
      return null;
    }
    
    // Determine if price trend is positive
    const isPositive = chartData.prices[0][1] <= chartData.prices[chartData.prices.length - 1][1];
    const lineColor = isPositive ? 'rgb(22, 199, 132)' : 'rgb(234, 57, 67)';
    const fillColor = isPositive ? 'rgba(22, 199, 132, 0.1)' : 'rgba(234, 57, 67, 0.1)';
    
    return {
      labels: chartData.prices.map(data => formatDate(data[0])),
      datasets: [
        {
          label: `Price (${currency.toUpperCase()})`,
          data: chartData.prices.map(data => data[1]),
          borderColor: lineColor,
          backgroundColor: fillColor,
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
        <button 
          className={timeRange === 1 ? 'active' : ''} 
          onClick={() => setTimeRange(1)}
        >
          24h
        </button>
        <button 
          className={timeRange === 7 ? 'active' : ''} 
          onClick={() => setTimeRange(7)}
        >
          7d
        </button>
        <button 
          className={timeRange === 30 ? 'active' : ''} 
          onClick={() => setTimeRange(30)}
        >
          30d
        </button>
        <button 
          className={timeRange === 90 ? 'active' : ''} 
          onClick={() => setTimeRange(90)}
        >
          90d
        </button>
        <button 
          className={timeRange === 365 ? 'active' : ''} 
          onClick={() => setTimeRange(365)}
        >
          1y
        </button>
      </div>

      <div className="chart-wrapper">
        {isLoading && <div className="chart-loading">Loading chart data...</div>}
        {error && <div className="chart-error">{error}</div>}
        {!isLoading && !error && data && (
          <Line options={chartOptions} data={data} height={80} ref={chartRef} />
        )}
      </div>
    </div>
  );
};

export default CoinChart;