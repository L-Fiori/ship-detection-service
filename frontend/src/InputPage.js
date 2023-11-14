// src/InputPage.js
import React, { useState } from 'react';
import './InputPage.css'; // Import the CSS file

const InputPage = ({ onSubmit }) => {
  const [region, setRegion] = useState('');
  const [dateRange, setDateRange] = useState('');
  const [cloudPercentage, setCloudPercentage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // Validate inputs here if needed
    onSubmit({ region, dateRange, cloudPercentage });
  };

  return (
    <div>
      <h1>Input Page</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Region:
          <input
            type="text"
            value={region}
            onChange={(e) => setRegion(e.target.value)}
            required
          />
        </label>
        <br />
        <label>
          Date Range:
          <input
            type="text"
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            required
          />
        </label>
        <br />
        <label>
          Cloud Percentage:
          <input
            type="text"
            value={cloudPercentage}
            onChange={(e) => setCloudPercentage(e.target.value)}
            required
          />
        </label>
        <br />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
};

export default InputPage;
