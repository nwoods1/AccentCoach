import React, { createContext, useContext, useEffect, useState } from "react";
import { onAuthStateChanged } from "firebase/auth";
import { auth, db } from "../../firebase/firebase"; 
import { doc, getDoc, setDoc, serverTimestamp } from "firebase/firestore";

const AuthContext = createContext({
  user: null,
  userLoggedIn: false,
  loading: true,
});

async function ensureUserDoc(user, totalLevels = 5) {
  try {
    const ref = doc(db, "users", user.uid);
    const snap = await getDoc(ref);

    if (!snap.exists()) {
      await setDoc(ref, {
        email: user.email ?? null,
        displayName: user.displayName ?? null,
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
        accentProgress: {
          levels: Array(totalLevels).fill(false), // [false, false, false, false, false]
          highestLevel: 0,
          completed: false,
          lastLevel: null,
          lastResult: null,
          lastConfidence: null,
          attemptsCount: 0,
        },
      });
    } else {
      await setDoc(ref, { updatedAt: serverTimestamp() }, { merge: true });
    }
  } catch (err) {
    console.error("ensureUserDoc error:", err);
  }
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const off = onAuthStateChanged(auth, (u) => {
      setUser(u || null);
      setLoading(false);
      if (u) {
        ensureUserDoc(u).catch((e) => console.error("ensureUserDoc error:", e));
      }      
    });
    return () => off();
  }, []);

  return (
    <AuthContext.Provider value={{ user, userLoggedIn: !!user, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
