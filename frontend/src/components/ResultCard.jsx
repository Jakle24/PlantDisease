import React, { useState, useEffect } from "react";
import MiniGame from "./MiniGame";
import axios from "axios";

export default function ResultCard({ result }) {
  const [showGame, setShowGame] = useState(false);
  const [canPlayGame, setCanPlayGame] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

  useEffect(() => {
    // Check if user has already played today
    async function checkDailyPlay() {
      try {
        const res = await axios.get(`${API_URL}/profile`);
        const lastPlay = res.data.last_game_play; // We'll track this in backend
        const today = new Date().toISOString().split("T")[0];
        setCanPlayGame(lastPlay !== today);
      } catch (err) {
        console.error("Failed to check daily game status:", err);
        setCanPlayGame(false);
      }
    }
    checkDailyPlay();
  }, [API_URL]);

  const handleGameEnd = async (score) => {
    alert(`You earned ${score} XP from the mini-game!`);
    setCanPlayGame(false);
    setShowGame(false);

    // POST score to backend to update XP
    try {
      await axios.post(`${API_URL}/add_game_xp`, { xp: score });
    } catch (err) {
      console.error("Failed to add game XP:", err);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 mt-4 max-w-md mx-auto border border-gray-200 dark:border-gray-700">
      {/* Disease Header */}
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-green-700 dark:text-green-400">
          {result.disease}
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mt-1">Prediction Result</p>
      </div>

      {/* Confidence Bar */}
      <div className="mb-4">
        <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
          Confidence: {result.confidence}
        </p>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
          <div
            className="bg-green-500 h-4 rounded-full transition-all duration-500"
            style={{ width: result.confidence }}
          />
        </div>
      </div>

      {/* Gamification Stats */}
      <div className="flex flex-wrap gap-2 mb-4">
        <div className="px-3 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-100 rounded-full text-sm font-medium">
          XP: {result.xpGained}
        </div>
        <div className="px-3 py-1 bg-yellow-100 dark:bg-yellow-800 text-yellow-800 dark:text-yellow-100 rounded-full text-sm font-medium">
          Streak: {result.streak} days
        </div>
        {result.badges?.map((badge, i) => (
          <div
            key={i}
            className="px-3 py-1 bg-purple-100 dark:bg-purple-800 text-purple-800 dark:text-purple-100 rounded-full text-sm font-medium"
          >
            {badge}
          </div>
        ))}
      </div>

      {/* Fact / Tip Section */}
      {result.fact && (
        <div className="mt-2 p-4 bg-green-50 dark:bg-green-900 rounded-xl shadow-inner">
          <p className="text-sm italic text-gray-700 dark:text-gray-200">
            💡 {result.fact}
          </p>
        </div>
      )}

      {/* Daily Mini-Game */}
      {canPlayGame ? (
        <>
          <button
            className="mt-4 px-4 py-2 bg-yellow-500 text-black rounded hover:bg-yellow-600"
            onClick={() => setShowGame(prev => !prev)}
          >
            {showGame ? "Close Mini-Game" : "Play Daily Mini-Game"}
          </button>
          {showGame && <MiniGame onScore={handleGameEnd} />}
        </>
      ) : (
        <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
          Daily mini-game already played today!
        </p>
      )}
    </div>
  );
}
