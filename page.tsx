'use client';

import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Calendar, Target, BarChart3, RefreshCw } from 'lucide-react';

export default function StockPredictor() {
  const [timeframe, setTimeframe] = useState('3months');
  const [category, setCategory] = useState('all');
  
  // Mock data for top 5 performing stocks from previous week
  const topStocks = [
    {
      symbol: 'RELIANCE',
      name: 'Reliance Industries',
      category: 'Energy',
      mondayOpen: 2845.50,
      fridayClose: 2987.20,
      return: 4.98,
      volume: '12.5M'
    },
    {
      symbol: 'TCS',
      name: 'Tata Consultancy Services',
      category: 'IT',
      mondayOpen: 3650.80,
      fridayClose: 3812.40,
      return: 4.43,
      volume: '8.2M'
    },
    {
      symbol: 'INFY',
      name: 'Infosys',
      category: 'IT',
      mondayOpen: 1456.30,
      fridayClose: 1518.90,
      return: 4.30,
      volume: '15.3M'
    },
    {
      symbol: 'HDFCBANK',
      name: 'HDFC Bank',
      category: 'Banking',
      mondayOpen: 1589.60,
      fridayClose: 1647.80,
      return: 3.66,
      volume: '18.7M'
    },
    {
      symbol: 'TATAMOTORS',
      name: 'Tata Motors',
      category: 'Auto',
      mondayOpen: 876.40,
      fridayClose: 907.50,
      return: 3.55,
      volume: '22.1M'
    }
  ];

  // Mock predictions for upcoming week
  const predictions = [
    {
      symbol: 'ADANIPORTS',
      name: 'Adani Ports',
      category: 'Infrastructure',
      currentPrice: 1245.80,
      predictedReturn: 5.2,
      confidence: 78,
      signals: ['Technical Breakout', 'Volume Surge', 'Positive News']
    },
    {
      symbol: 'BAJFINANCE',
      name: 'Bajaj Finance',
      category: 'Financial Services',
      currentPrice: 6789.30,
      predictedReturn: 4.8,
      confidence: 82,
      signals: ['Strong Fundamentals', 'RSI Oversold', 'Sector Momentum']
    },
    {
      symbol: 'ASIANPAINT',
      name: 'Asian Paints',
      category: 'Consumer Goods',
      currentPrice: 3012.50,
      predictedReturn: 4.1,
      confidence: 75,
      signals: ['Earnings Beat', 'Moving Avg Cross', 'Institutional Buying']
    },
    {
      symbol: 'SUNPHARMA',
      name: 'Sun Pharmaceutical',
      category: 'Pharma',
      currentPrice: 1456.90,
      predictedReturn: 3.9,
      confidence: 71,
      signals: ['FDA Approval', 'Chart Pattern', 'Export Growth']
    },
    {
      symbol: 'MARUTI',
      name: 'Maruti Suzuki',
      category: 'Auto',
      currentPrice: 11234.60,
      predictedReturn: 3.5,
      confidence: 69,
      signals: ['Sales Growth', 'Support Level', 'Demand Recovery']
    }
  ];

  // Mock performance data
  const performanceData = {
    '1month': { accuracy: 68.5, totalPredictions: 20, profitable: 14, avgReturn: 2.3 },
    '3months': { accuracy: 72.3, totalPredictions: 60, profitable: 44, avgReturn: 2.8 },
    '6months': { accuracy: 69.8, totalPredictions: 120, profitable: 84, avgReturn: 2.5 },
    '1year': { accuracy: 71.2, totalPredictions: 240, profitable: 171, avgReturn: 2.7 }
  };

  const categoryPerformance = [
    { category: 'IT', accuracy: 75.4, predictions: 45, avgReturn: 3.2 },
    { category: 'Banking', accuracy: 71.2, predictions: 52, avgReturn: 2.8 },
    { category: 'Auto', accuracy: 68.9, predictions: 38, avgReturn: 2.4 },
    { category: 'Pharma', accuracy: 73.1, predictions: 41, avgReturn: 3.0 },
    { category: 'Energy', accuracy: 69.5, predictions: 35, avgReturn: 2.6 }
  ];

  const currentPerf = performanceData[timeframe];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">NSE Stock Predictor</h1>
          <p className="text-blue-200">AI-Powered Weekly Stock Performance Analysis & Predictions</p>
          <div className="flex items-center gap-2 mt-3 text-sm text-blue-300">
            <Calendar className="w-4 h-4" />
            <span>Last Updated: Monday, Oct 18, 2025 - 09:15 AM IST</span>
          </div>
        </div>

        {/* Top Performers - Previous Week */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 mb-6 border border-white/20">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-6 h-6 text-green-400" />
            <h2 className="text-2xl font-bold text-white">Top 5 Performers - Previous Week</h2>
          </div>
          <p className="text-blue-200 text-sm mb-4">Monday Oct 11 (Open) → Friday Oct 15 (Close)</p>
          
          <div className="space-y-3">
            {topStocks.map((stock, idx) => (
              <div key={stock.symbol} className="bg-white/5 rounded-lg p-4 border border-white/10 hover:bg-white/10 transition">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-2xl font-bold text-blue-400">#{idx + 1}</span>
                      <div>
                        <h3 className="text-lg font-bold text-white">{stock.symbol}</h3>
                        <p className="text-sm text-blue-300">{stock.name}</p>
                      </div>
                      <span className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-full text-xs font-semibold">
                        {stock.category}
                      </span>
                    </div>
                    <div className="grid grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-blue-300">Monday Open</p>
                        <p className="text-white font-semibold">₹{stock.mondayOpen.toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-blue-300">Friday Close</p>
                        <p className="text-white font-semibold">₹{stock.fridayClose.toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-blue-300">Volume</p>
                        <p className="text-white font-semibold">{stock.volume}</p>
                      </div>
                      <div>
                        <p className="text-blue-300">Weekly Return</p>
                        <p className={`text-2xl font-bold ${stock.return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {stock.return >= 0 ? '+' : ''}{stock.return.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Predictions for Upcoming Week */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 mb-6 border border-white/20">
          <div className="flex items-center gap-2 mb-4">
            <Target className="w-6 h-6 text-purple-400" />
            <h2 className="text-2xl font-bold text-white">AI Predictions - Upcoming Week</h2>
          </div>
          <p className="text-blue-200 text-sm mb-4">Predicted top performers for Oct 18-22, 2025</p>
          
          <div className="space-y-3">
            {predictions.map((pred, idx) => (
              <div key={pred.symbol} className="bg-white/5 rounded-lg p-4 border border-white/10 hover:bg-white/10 transition">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl font-bold text-purple-400">#{idx + 1}</span>
                    <div>
                      <h3 className="text-lg font-bold text-white">{pred.symbol}</h3>
                      <p className="text-sm text-blue-300">{pred.name}</p>
                    </div>
                    <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-xs font-semibold">
                      {pred.category}
                    </span>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-blue-300">Current Price</p>
                    <p className="text-lg font-bold text-white">₹{pred.currentPrice.toFixed(2)}</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4 mb-3">
                  <div>
                    <p className="text-blue-300 text-sm">Predicted Return</p>
                    <p className="text-2xl font-bold text-green-400">+{pred.predictedReturn.toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-blue-300 text-sm">Confidence</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-white/10 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-yellow-400 to-green-400 h-2 rounded-full transition-all"
                          style={{ width: `${pred.confidence}%` }}
                        />
                      </div>
                      <span className="text-white font-semibold">{pred.confidence}%</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-blue-300 text-sm">Target Price</p>
                    <p className="text-lg font-bold text-white">₹{(pred.currentPrice * (1 + pred.predictedReturn/100)).toFixed(2)}</p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {pred.signals.map((signal, i) => (
                    <span key={i} className="px-2 py-1 bg-green-500/20 text-green-300 rounded text-xs">
                      {signal}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Model Performance */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
          <div className="flex items-center gap-2 mb-6">
            <BarChart3 className="w-6 h-6 text-yellow-400" />
            <h2 className="text-2xl font-bold text-white">Predictive Model Performance</h2>
          </div>

          {/* Timeframe Selector */}
          <div className="flex gap-2 mb-6">
            {['1month', '3months', '6months', '1year'].map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-4 py-2 rounded-lg font-semibold transition ${
                  timeframe === tf
                    ? 'bg-yellow-500 text-slate-900'
                    : 'bg-white/10 text-white hover:bg-white/20'
                }`}
              >
                {tf === '1month' ? '1 Month' : tf === '3months' ? '3 Months' : tf === '6months' ? '6 Months' : '1 Year'}
              </button>
            ))}
          </div>

          {/* Aggregate Performance */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <p className="text-blue-300 text-sm mb-1">Accuracy Rate</p>
              <p className="text-3xl font-bold text-green-400">{currentPerf.accuracy}%</p>
            </div>
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <p className="text-blue-300 text-sm mb-1">Total Predictions</p>
              <p className="text-3xl font-bold text-white">{currentPerf.totalPredictions}</p>
            </div>
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <p className="text-blue-300 text-sm mb-1">Profitable Picks</p>
              <p className="text-3xl font-bold text-green-400">{currentPerf.profitable}</p>
            </div>
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <p className="text-blue-300 text-sm mb-1">Avg Return</p>
              <p className="text-3xl font-bold text-yellow-400">+{currentPerf.avgReturn}%</p>
            </div>
          </div>

          {/* Category Performance */}
          <h3 className="text-lg font-bold text-white mb-4">Performance by Category</h3>
          <div className="space-y-3">
            {categoryPerformance.map((cat) => (
              <div key={cat.category} className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-white font-semibold">{cat.category}</h4>
                  <span className="text-sm text-blue-300">{cat.predictions} predictions</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-blue-300 text-sm">Accuracy</p>
                    <p className="text-xl font-bold text-green-400">{cat.accuracy}%</p>
                  </div>
                  <div>
                    <p className="text-blue-300 text-sm">Avg Return</p>
                    <p className="text-xl font-bold text-yellow-400">+{cat.avgReturn}%</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
          <p className="text-red-300 text-sm">
            <strong>Disclaimer:</strong> This is a demo application with mock data. Stock predictions are for educational purposes only. 
            Always do your own research and consult with financial advisors before making investment decisions. Past performance does not guarantee future results.
          </p>
        </div>
      </div>
    </div>
  );
};

export default StockPredictor;
