import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/authContext';
import { doSignOut } from '../firebase/auth';
import Parrot from '../img/mascot-parrot.png';

const NAV_LINKS = [
  { to: '/home', label: 'Home' },
  { to: '/practice', label: 'Practice' },
  { to: '/analyze', label: 'Analyze' },
];

const Header = () => {
  const { userLoggedIn, user } = useAuth();
  const loc = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);

  const onLogout = async () => {
    try {
      await doSignOut();
    } catch (e) {
      console.error(e);
    }
  };

  const initials = (user?.displayName || user?.email || '?').trim().charAt(0).toUpperCase();

  return (
    <header className="fixed top-0 inset-x-0 z-40 h-16 border-b border-slate-800/60 bg-slate-900/90 backdrop-blur text-slate-100 shadow-sm">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-full flex items-center justify-between gap-4">
        <div className="flex items-center gap-6 min-w-0">
          <Link
            to={userLoggedIn ? '/home' : '/login'}
            className="flex items-center gap-2 shrink-0 rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-400"
          >
            <img
              src={Parrot}
              alt="Accent Coach mascot"
              className="h-9 w-9 rounded-full object-cover object-top ring-2 ring-brand-400/70 bg-slate-800"
              loading="eager"
            />
            <span className="font-bold text-base sm:text-lg tracking-tight">Accent Coach</span>
          </Link>

          {userLoggedIn && (
            <nav className="hidden sm:flex items-center gap-1 text-sm">
              {NAV_LINKS.map((link) => {
                const active = loc.pathname === link.to;
                return (
                  <Link
                    key={link.to}
                    to={link.to}
                    className={[
                      'px-3 py-1.5 rounded-lg font-medium transition-colors',
                      active
                        ? 'bg-white/10 text-white'
                        : 'text-slate-300 hover:text-white hover:bg-white/5',
                    ].join(' ')}
                  >
                    {link.label}
                  </Link>
                );
              })}
            </nav>
          )}
        </div>

        <div className="flex items-center gap-2 sm:gap-3">
          {userLoggedIn ? (
            <>
              <div className="hidden sm:flex items-center gap-2 pl-1">
                <span className="h-7 w-7 rounded-full bg-brand-500/90 text-slate-900 text-xs font-bold flex items-center justify-center">
                  {initials}
                </span>
                <span className="text-sm text-slate-300 max-w-[10rem] truncate">
                  {user?.displayName || user?.email}
                </span>
              </div>
              <button
                onClick={onLogout}
                className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-sm font-medium transition-colors"
                aria-label="Log out"
              >
                Log out
              </button>
              <button
                type="button"
                onClick={() => setMenuOpen((v) => !v)}
                className="sm:hidden inline-flex items-center justify-center h-9 w-9 rounded-lg hover:bg-white/10 transition-colors"
                aria-label="Toggle menu"
                aria-expanded={menuOpen}
              >
                <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2}>
                  {menuOpen ? (
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 6l12 12M18 6l-12 12" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 7h16M4 12h16M4 17h16" />
                  )}
                </svg>
              </button>
            </>
          ) : (
            <Link
              to="/login"
              className="px-3 py-1.5 rounded-lg bg-brand-500 hover:bg-brand-600 text-slate-900 text-sm font-semibold transition-colors"
            >
              Log in
            </Link>
          )}
        </div>
      </div>

      {userLoggedIn && menuOpen && (
        <nav className="sm:hidden fixed top-16 inset-x-0 z-40 border-t border-slate-800 bg-slate-900 px-4 py-2 flex flex-col shadow-lg">
          {NAV_LINKS.map((link) => {
            const active = loc.pathname === link.to;
            return (
              <Link
                key={link.to}
                to={link.to}
                onClick={() => setMenuOpen(false)}
                className={[
                  'px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                  active ? 'bg-white/10 text-white' : 'text-slate-300 hover:text-white hover:bg-white/5',
                ].join(' ')}
              >
                {link.label}
              </Link>
            );
          })}
          <div className="mt-1 pt-2 border-t border-slate-800 text-xs text-slate-400 px-3 pb-1 truncate">
            Signed in as {user?.displayName || user?.email}
          </div>
        </nav>
      )}
    </header>
  );
};

export default Header;
