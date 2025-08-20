import React, { useEffect, useState } from "react";

export default function ResultCard({ result }) {
  const [animatedXp, setAnimatedXp] = useState(0);
  const [glow, setGlow] = useState(false);

  const confidenceValue = parseFloat(result.confidence);
  let confidenceColor = "bg-red-500";
  if (confidenceValue >= 75) confidenceColor = "bg-green-500";
  else if (confidenceValue >= 50) confidenceColor = "bg-yellow-400";

  // Animate XP count up
  useEffect(() => {
    setAnimatedXp(0);
    let start = 0;
    const end = result.xpGained;
    if (end > 0) {
      const duration = 800; // ms
      const stepTime = Math.max(Math.floor(duration / end), 20);
      const timer = setInterval(() => {
        start += 1;
        setAnimatedXp(start);
        if (start >= end) clearInterval(timer);
      }, stepTime);
    }
    // trigger glow effect for streak
    setGlow(true);
    const glowTimer = setTimeout(() => setGlow(false), 1000);

    return () => clearInterval(glowTimer);
  }, [result.xpGained, result.streak]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-6 mt-4">
      <h2 className="text-2xl font-bold mb-4">Prediction Result</h2>

      <p className="mb-2">
        <strong>Disease:</strong> {result.disease}
      </p>

      <p className="mb-1">
        <strong>Confidence:</strong> {result.confidence}%
      </p>
      <div className="w-full h-4 rounded-full bg-gray-300 dark:bg-gray-700 mb-3 overflow-hidden">
        <div
          className={`${confidenceColor} h-4`}
          style={{ width: `${confidenceValue}%`, transition: "width 0.5s ease" }}
        />
      </div>

      <p className="mb-2">
        <strong>XP Gained:</strong> {animatedXp}
      </p>
      <p
        className={`mb-2 transition-all duration-500 ${
          glow ? "text-yellow-400 font-bold scale-105" : ""
        }`}
      >
        <strong>Current Streak:</strong> {result.streak} days
      </p>

      {result.badges && result.badges.length > 0 && (
        <div className="mt-3">
          <strong>Badges:</strong>
          <div className="flex flex-wrap gap-2 mt-1">
            {result.badges.map((badge, i) => (
              <span
                key={i}
                className="bg-blue-200 dark:bg-blue-700 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full text-sm font-medium"
              >
                {badge}
              </span>
            ))}
          </div>
        </div>
      )}

      {result.fact && (
        <div className="mt-4 p-3 bg-green-50 dark:bg-green-900 rounded-xl shadow-inner">
          <p className="text-sm italic text-gray-700 dark:text-gray-200">💡 {result.fact}</p>
        </div>
      )}
    </div>
  );
}
