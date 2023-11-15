import React, { useState } from 'react';

const InputPage = ({ onSubmit }) => {
  const [loc, setLoc] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [cloud, setCloud] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('http://127.0.0.1:5000/collect_inputs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json', // Set appropriate content type
        },
        body: JSON.stringify({ loc, startDate, endDate, cloud }), // Convert form data to JSON
      });

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
