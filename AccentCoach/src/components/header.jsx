import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/authContext';
import { doSignOut } from '../firebase/auth';
import Parrot from '../img/mascot-parrot.png';

const Header = () => {
  const { userLoggedIn, user } = useAuth();
  const loc = useLocation();

  const onLogout = async () => {
    try {
      await doSignOut();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <header className="fixed top-0 inset-x-0 z-40 backdrop-blur bg-slate-900/60 text-slate-100">
      {/* increased height */}
      <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to={userLoggedIn ? '/home' : '/login'} className="flex items-center gap-3">
            <img
              src={Parrot}
              alt="Accent Coach mascot"
              className="h-12 md:h-20 w-auto drop-shadow"
              loading="eager"
            />
            <span className="font-semibold text-lg md:text-xl">Accent Coach</span>
          </Link>

          {userLoggedIn && (
            <nav className="ml-2 hidden sm:flex gap-5 text-sm text-slate-300">
              <Link
                className={loc.pathname === '/home' ? 'text-white font-medium' : 'hover:text-white'}
                to="/home"
              >
                Home
              </Link>
              <Link
                className={loc.pathname === '/practice' ? 'text-white font-medium' : 'hover:text-white'}
                to="/practice"
              >
                Practice
              </Link>
            </nav>
          )}
        </div>

        <div className="flex items-center gap-3">
          {userLoggedIn ? (
            <>
              <span className="hidden sm:block text-sm text-slate-300">
                {user?.displayName || user?.email}
              </span>
              <button
                onClick={onLogout}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm"
                aria-label="Log out"
              >
                Log out
              </button>
            </>
          ) : (
            <Link
              to="/login"
              className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-sm"
            >
              Log in
            </Link>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
