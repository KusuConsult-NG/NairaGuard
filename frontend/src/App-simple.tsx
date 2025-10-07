import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';

// Simple Header Component
const Header = () => (
  <header style={{ backgroundColor: '#f8f9fa', padding: '1rem', borderBottom: '1px solid #dee2e6' }}>
    <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ fontSize: '1.5rem' }}>🛡️</span>
        <span style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>NairaGuard</span>
      </div>
      <nav style={{ display: 'flex', gap: '1rem' }}>
        <Link to="/" style={{ textDecoration: 'none', color: '#495057' }}>Home</Link>
        <Link to="/about" style={{ textDecoration: 'none', color: '#495057' }}>About</Link>
        <Link to="/admin" style={{ textDecoration: 'none', color: '#495057' }}>Admin</Link>
      </nav>
    </div>
  </header>
);

// Simple Home Page
const HomePage = () => (
  <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
    <h1 style={{ fontSize: '2rem', marginBottom: '1rem' }}>NairaGuard</h1>
    <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: '#6c757d' }}>
      Welcome to NairaGuard - Your AI-powered fake Naira detection system.
    </p>
    <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#e9ecef', borderRadius: '0.5rem' }}>
      <h3>Features:</h3>
      <ul>
        <li>AI-powered currency detection</li>
        <li>Real-time analysis</li>
        <li>Admin dashboard</li>
        <li>Secure verification</li>
      </ul>
    </div>
  </div>
);

// Simple About Page
const AboutPage = () => (
  <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
    <h1 style={{ fontSize: '2rem', marginBottom: '1rem' }}>About NairaGuard</h1>
    <p style={{ fontSize: '1.1rem', lineHeight: '1.6', color: '#6c757d' }}>
      NairaGuard is an advanced AI system designed to detect counterfeit Nigerian Naira notes.
    </p>
    <div style={{ marginTop: '2rem' }}>
      <h3>How it works:</h3>
      <ol>
        <li>Upload an image of a Naira note</li>
        <li>Our AI analyzes the security features</li>
        <li>Get instant results with confidence scores</li>
        <li>View detailed analysis and recommendations</li>
      </ol>
    </div>
  </div>
);

// Simple Admin Page
const AdminPage = () => (
  <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
    <h1 style={{ fontSize: '2rem', marginBottom: '1rem' }}>Admin Dashboard</h1>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '2rem' }}>
      <div style={{ padding: '1rem', backgroundColor: '#e9ecef', borderRadius: '0.5rem' }}>
        <h3>Total Detections</h3>
        <p style={{ fontSize: '2rem', fontWeight: 'bold', color: '#007bff' }}>1,234</p>
      </div>
      <div style={{ padding: '1rem', backgroundColor: '#e9ecef', borderRadius: '0.5rem' }}>
        <h3>Genuine Notes</h3>
        <p style={{ fontSize: '2rem', fontWeight: 'bold', color: '#28a745' }}>1,100</p>
      </div>
      <div style={{ padding: '1rem', backgroundColor: '#e9ecef', borderRadius: '0.5rem' }}>
        <h3>Fake Notes</h3>
        <p style={{ fontSize: '2rem', fontWeight: 'bold', color: '#dc3545' }}>134</p>
      </div>
    </div>
  </div>
);

// Main App Component
const App: React.FC = () => {
  return (
    <Router>
      <div style={{ minHeight: '100vh', backgroundColor: '#ffffff' }}>
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/admin" element={<AdminPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
