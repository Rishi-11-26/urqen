-- ============================================================================
-- PREVIA PLATFORM — Supabase PostgreSQL Database & Storage Schema
-- ============================================================================
-- Safe & Idempotent: You can run this script repeatedly without any errors.
-- ============================================================================

-- 1. Enable UUID Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Datasets Metadata Table
CREATE TABLE IF NOT EXISTS public.datasets (
    dataset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    size_bytes BIGINT DEFAULT 0,
    n_rows INT DEFAULT 0,
    n_columns INT DEFAULT 0,
    columns JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (RLS) on Datasets
ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can only view their own datasets" ON public.datasets;
CREATE POLICY "Users can only view their own datasets"
    ON public.datasets FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can only insert their own datasets" ON public.datasets;
CREATE POLICY "Users can only insert their own datasets"
    ON public.datasets FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can only delete their own datasets" ON public.datasets;
CREATE POLICY "Users can only delete their own datasets"
    ON public.datasets FOR DELETE
    USING (auth.uid() = user_id);

-- 3. Experiments Metadata & Leaderboard Table
CREATE TABLE IF NOT EXISTS public.experiments (
    experiment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES public.datasets(dataset_id) ON DELETE CASCADE,
    dataset_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    target_column TEXT NOT NULL,
    mode TEXT DEFAULT 'guided',
    status TEXT DEFAULT 'pending',
    best_model TEXT,
    winner_explanation TEXT,
    primary_metric TEXT DEFAULT 'f1',
    models JSONB DEFAULT '[]'::jsonb,
    results JSONB DEFAULT '{}'::jsonb,
    configuration JSONB DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Enable RLS on Experiments
ALTER TABLE public.experiments ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can only view their own experiments" ON public.experiments;
CREATE POLICY "Users can only view their own experiments"
    ON public.experiments FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can only insert their own experiments" ON public.experiments;
CREATE POLICY "Users can only insert their own experiments"
    ON public.experiments FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can only delete their own experiments" ON public.experiments;
CREATE POLICY "Users can only delete their own experiments"
    ON public.experiments FOR DELETE
    USING (auth.uid() = user_id);

-- 4. User Storage Quotas Table
CREATE TABLE IF NOT EXISTS public.user_storage_quotas (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    dataset_bytes BIGINT DEFAULT 0,
    artifact_bytes BIGINT DEFAULT 0,
    quota_limit_bytes BIGINT DEFAULT 52428800, -- 50 MB default
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.user_storage_quotas ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own quota" ON public.user_storage_quotas;
CREATE POLICY "Users can view their own quota"
    ON public.user_storage_quotas FOR SELECT
    USING (auth.uid() = user_id);

-- 5. Storage Buckets for Raw Datasets and Model Artifacts
INSERT INTO storage.buckets (id, name, public)
VALUES ('previa-datasets', 'previa-datasets', false)
ON CONFLICT (id) DO NOTHING;

INSERT INTO storage.buckets (id, name, public)
VALUES ('previa-artifacts', 'previa-artifacts', false)
ON CONFLICT (id) DO NOTHING;

-- Storage RLS Policies (Safe drop & create)
DROP POLICY IF EXISTS "Users access own dataset files" ON storage.objects;
CREATE POLICY "Users access own dataset files"
    ON storage.objects FOR ALL
    USING (bucket_id = 'previa-datasets' AND (storage.foldername(name))[1] = auth.uid()::text);

DROP POLICY IF EXISTS "Users access own artifact files" ON storage.objects;
CREATE POLICY "Users access own artifact files"
    ON storage.objects FOR ALL
    USING (bucket_id = 'previa-artifacts' AND (storage.foldername(name))[1] = auth.uid()::text);
