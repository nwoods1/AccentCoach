import React, { JSX } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import Header from './components/header';
import Login from './components/Login';
import Register from './components/register';
import Recorder from './components/Recorder';
import { useAuth } from './contexts/authContext';
import PhonemeAnalyzer from './components/PhonemeAnalyzer';
import Home from './components/Home'

function Protected({ children }: { children: JSX.Element }) {
  const { userLoggedIn, loading } = useAuth();
  if (loading) return <div className="p-6">Loading…</div>;
  return userLoggedIn ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const { userLoggedIn } = useAuth();
  const { pathname } = useLocation();
  const onAuthPage = pathname === '/login' || pathname === '/register';

  return (
    <>
      {userLoggedIn && !onAuthPage && <Header />}

      <div className={userLoggedIn && !onAuthPage ? 'pt-12' : ''}>
        <Routes>
           <Route path="/" element={userLoggedIn ? <Navigate to="/home" replace /> : <Navigate to="/login" replace />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/home" element={
            <Protected>
              <Home />
            </Protected>
          } />

          <Route path="/practice" element={
            <Protected>
              <div className="max-w-3xl mx-auto p-4">
                <Recorder />
              </div>
            </Protected>
          } />
          
          <Route path="*" element={<Navigate to="/" replace />} />
          <Route
            path="/analyze"
            element={
              <Protected>
                <PhonemeAnalyzer />
              </Protected>
            }
          />
        </Routes>
      </div>
    </>
  );
}
