import React from "react";

export default function ResultCard({ result }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-md p-4 mt-4">
      <h2 className="text-xl font-bold mb-2">Prediction Result</h2>
      <p><strong>Disease:</strong> {result.disease}</p>
      <p><strong>Confidence:</strong> {result.confidence}%</p>
      <p><strong>XP Gained:</strong> {result.xp}</p>
      <p><strong>Current Streak:</strong> {result.streak} days</p>

      {result.badges && result.badges.length > 0 && (
        <div className="mt-2">
          <strong>Badges:</strong>
          <ul className="list-disc ml-6">
            {result.badges.map((badge, i) => (
              <li key={i}>{badge}</li>
            ))}
          </ul>
        </div>
      )}
      {result.fact && (
  <div className="mt-3 p-3 bg-green-50 dark:bg-green-900 rounded-xl shadow-inner">
    <p className="text-sm italic text-gray-700 dark:text-gray-200">
      💡 {result.fact}
    </p>
  </div>
)}

    </div>
  );
}
