-- Enable Row Level Security (RLS) on all tables for pipeline isolation
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipelines ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_pairs ENABLE ROW LEVEL SECURITY;
ALTER TABLE finetune_jobs ENABLE ROW LEVEL SECURITY;

-- Force RLS even for table owners to ensure strict pipeline isolation
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE pipelines FORCE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs FORCE ROW LEVEL SECURITY;
ALTER TABLE evaluations FORCE ROW LEVEL SECURITY;
ALTER TABLE training_pairs FORCE ROW LEVEL SECURITY;
ALTER TABLE finetune_jobs FORCE ROW LEVEL SECURITY;

-- Drop existing policies if any
DROP POLICY IF EXISTS documents_pipeline_isolation ON documents;
DROP POLICY IF EXISTS chunks_pipeline_isolation ON chunks;
DROP POLICY IF EXISTS pipelines_pipeline_isolation ON pipelines;
DROP POLICY IF EXISTS pipeline_runs_pipeline_isolation ON pipeline_runs;
DROP POLICY IF EXISTS evaluations_pipeline_isolation ON evaluations;
DROP POLICY IF EXISTS training_pairs_pipeline_isolation ON training_pairs;

-- Define RLS policies based on PostgreSQL session variable 'app.current_pipeline_id'
CREATE POLICY documents_pipeline_isolation ON documents
  FOR ALL
  USING (pipeline_id IS NOT NULL AND pipeline_id = NULLIF(current_setting('app.current_pipeline_id', true), '')::uuid);

CREATE POLICY chunks_pipeline_isolation ON chunks
  FOR ALL
  USING (
    document_id IN (
      SELECT id FROM documents
      WHERE pipeline_id IS NOT NULL AND pipeline_id = NULLIF(current_setting('app.current_pipeline_id', true), '')::uuid
    )
  );

CREATE POLICY pipelines_pipeline_isolation ON pipelines
  FOR ALL
  USING (id = NULLIF(current_setting('app.current_pipeline_id', true), '')::uuid);

CREATE POLICY pipeline_runs_pipeline_isolation ON pipeline_runs
  FOR ALL
  USING (pipeline_id = NULLIF(current_setting('app.current_pipeline_id', true), '')::uuid);

CREATE POLICY evaluations_pipeline_isolation ON evaluations
  FOR ALL
  USING (
    run_id IN (
      SELECT id FROM pipeline_runs
      WHERE pipeline_id = NULLIF(current_setting('app.current_pipeline_id', true), '')::uuid
    )
  );

CREATE POLICY training_pairs_pipeline_isolation ON training_pairs
  FOR ALL
  USING (
    run_id IN (
      SELECT id FROM pipeline_runs
      WHERE pipeline_id = NULLIF(current_setting('app.current_pipeline_id', true), '')::uuid
    )
  );
