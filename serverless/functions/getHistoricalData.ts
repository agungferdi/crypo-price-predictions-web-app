import { APIGatewayProxyEvent, APIGatewayProxyResult } from 'aws-lambda';
import { coinGeckoAPI } from '../lib/coinGeckoApi';

/**
 * Lambda function to retrieve historical price data for a cryptocurrency
 */
export const handler = async (
  event: APIGatewayProxyEvent
): Promise<APIGatewayProxyResult> => {
  // Set up CORS headers
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Credentials': true,
    'Content-Type': 'application/json'
  };

  try {
    // Extract parameters from the event
    const coinId = event.pathParameters?.coinId;
    const currency = (event.queryStringParameters?.currency || 'usd').toLowerCase();
    const days = parseInt(event.queryStringParameters?.days || '180');

    if (!coinId) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({
          error: 'Missing required parameter: coinId'
        })
      };
    }

    // Validate days parameter
    if (isNaN(days) || days <= 0 || days > 365) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({
          error: 'Invalid days parameter. Must be a number between 1 and 365.'
        })
      };
    }

    // Get historical data from CoinGecko
    const historicalData = await coinGeckoAPI.getCoinHistoricalData(
      coinId, 
      days, 
      currency
    );

    // Return successful response
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        success: true,
        data: historicalData
      })
    };
  } catch (error) {
    console.error('Error fetching historical data:', error);
    
    // Return error response
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        success: false,
        error: 'Failed to fetch historical data',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    };
  }
};