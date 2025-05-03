import { APIGatewayProxyEvent, APIGatewayProxyResult } from 'aws-lambda';
import { coinGeckoAPI } from '../lib/coinGeckoApi';
import { PriceModel } from '../lib/priceModel';

/**
 * Lambda function to predict cryptocurrency prices using ML model
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

  // Handle OPTIONS requests for CORS
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  try {
    // Parse the request body
    const requestBody = event.body ? JSON.parse(event.body) : {};
    const { coinId, currency = 'usd', days = 180 } = requestBody;

    if (!coinId) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({
          success: false,
          error: 'Missing required parameter: coinId'
        })
      };
    }

    // Fetch historical data for training the model
    const historicalData = await coinGeckoAPI.getCoinHistoricalData(
      coinId,
      days,
      currency
    );

    if (!historicalData || !historicalData.prices || historicalData.prices.length < 60) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({
          success: false,
          error: 'Insufficient historical data for prediction'
        })
      };
    }

    // Initialize the price prediction model
    const priceModel = new PriceModel();
    
    // Train model and make predictions
    console.log(`Starting price prediction for ${coinId} in ${currency}`);
    const predictionResult = await priceModel.trainAndPredict(historicalData);
    console.log(`Prediction completed for ${coinId}`);

    // Get current price for reference
    const currentPrice = historicalData.prices[historicalData.prices.length - 1][1];

    // Calculate change percentages
    const changePercentages: Record<string, string> = {};
    for (const [timeFrame, predictedPrice] of Object.entries(predictionResult.predictions)) {
      const change = ((predictedPrice - currentPrice) / currentPrice) * 100;
      changePercentages[timeFrame] = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
    }

    // Return successful response with predictions
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        success: true,
        data: {
          predictions: predictionResult.predictions,
          confidence: predictionResult.confidence,
          metrics: predictionResult.metrics,
          currentPrice,
          changePercentages,
          currency
        }
      })
    };
  } catch (error) {
    console.error('Error predicting price:', error);
    
    // Return error response
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        success: false,
        error: 'Failed to predict price',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    };
  }
};