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

export const supabase = createClient(
  SUPABASE_URL || 'http://localhost',
  SUPABASE_ANON_KEY || 'local',
  {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
    },
  }
);
