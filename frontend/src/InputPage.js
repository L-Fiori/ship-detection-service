// src/InputPage.js
import React, { useState } from 'react';
import './InputPage.css'; // Import the CSS file

const InputPage = ({ onSubmit }) => {
  const [loc, setLoc] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [cloud, setCloud] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // Validate inputs here if needed
    onSubmit({ loc, startDate, endDate, cloud });
  };

  return (
    <div>
      <h1>Input Page</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Location:
          <input
            type="text"
            value={loc}
            onChange={(e) => setLoc(e.target.value)}
            required
          />
        </label>
        <br />
        <label>
          Start Date:
          <input
            type="text"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            required
          />
        </label>
        <br />
        <label>
          Cloud:
          <input
            type="text"
            value={cloud}
            onChange={(e) => setCloud(e.target.value)}
            required
          />
        </label>
        <br />
        <label>
          End Date:
          <input
            type="text"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
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
