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
  if (loading) {
    return (
      <div className="page flex items-center justify-center">
        <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm">
          <span className="h-4 w-4 rounded-full border-2 border-slate-300 border-t-brand-500 animate-spin" />
          Loading…
        </div>
      </div>
    );
  }
  return userLoggedIn ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const { userLoggedIn } = useAuth();
  const { pathname } = useLocation();
  const onAuthPage = pathname === '/login' || pathname === '/register';

  return (
    <>
      {userLoggedIn && !onAuthPage && <Header />}

      <div className={userLoggedIn && !onAuthPage ? 'pt-16' : ''}>
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
              <Recorder />
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
