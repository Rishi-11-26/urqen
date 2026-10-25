import { createClient } from '@supabase/supabase-js';

const configuredSupabaseUrl = import.meta.env.VITE_SUPABASE_URL?.trim().replace(/\/+$/, '');
const SUPABASE_URL = configuredSupabaseUrl?.replace(/\/(?:rest\/v1|auth\/v1)$/i, '');
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (configuredSupabaseUrl && SUPABASE_URL !== configuredSupabaseUrl) {
  console.warn('VITE_SUPABASE_URL should be the project URL; an API path suffix was removed.');
}

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.warn('Supabase credentials not configured. Running in local mode.');
}

export const isSupabaseConfigured = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

export const supabase = isSupabaseConfigured
  ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
      },
    })
  : {
      auth: {
        getSession: async () => ({ data: { session: null }, error: null }),
        onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
        signInWithPassword: async () => {
          throw new Error('Supabase authentication is not configured. Please provide VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
        },
        signUp: async () => {
          throw new Error('Supabase authentication is not configured. Please provide VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
        },
        signOut: async () => ({ error: null }),
      },
    };
