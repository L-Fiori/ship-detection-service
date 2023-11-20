import React, { useState } from 'react';
import { useNavigate } from "react-router-dom";
import './InputPage.css';

const InputPage = ({ onSubmit }) => {
  const navigate = useNavigate();
  const [loc, setLoc] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [cloud, setCloud] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('http://127.0.0.1:8080/collect_inputs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json', // Set appropriate content type
        },
        body: JSON.stringify({ loc, startDate, endDate, cloud }), // Convert form data to JSON
      });
      navigate("/processing");

      if (response.ok) {
        console.log('Data submitted successfully!');
        // Handle success or redirect here
      } else {
        console.error('Error submitting data.');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="input-container">
      <h1>Input Page</h1>
      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <label>Location:</label>
          <input
            type="text"
            value={loc}
            onChange={(e) => setLoc(e.target.value)}
            required
          />
        </div>
        <div className="input-group">
          <label>Start Date:</label>
          <input
            type="text"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            required
          />
        </div>
        <div className="input-group">
          <label>Cloud:</label>
          <input
            type="text"
            value={cloud}
            onChange={(e) => setCloud(e.target.value)}
            required
          />
        </div>
        <div className="input-group">
          <label>End Date:</label>
          <input
            type="text"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            required
          />
        </div>
        <button type="submit">Submit</button>
      </form>
    </div>
  );
};

export default InputPage;
