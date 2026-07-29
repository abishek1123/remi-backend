-- Quiz feature: record each answered question (topic + correctness) so the
-- learning graph can show per-topic mastery. Questions themselves are generated
-- on the fly and not persisted in v1 — only the outcomes that drive learning.

create table quiz_attempts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) not null,
  document_id uuid references documents(id) on delete set null,
  topic text not null,
  is_correct boolean not null,
  answered_at timestamp with time zone default now()
);

alter table quiz_attempts enable row level security;

create policy "Users manage own quiz attempts"
on quiz_attempts for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create index quiz_attempts_user_topic_idx on quiz_attempts (user_id, topic);

-- Per-topic mastery for the learning graph. Weakest topics first (feeds future
-- spaced repetition).
create or replace function topic_mastery(match_user_id uuid)
returns table (topic text, correct bigint, total bigint, mastery float)
language sql stable
as $$
  select
    topic,
    sum(case when is_correct then 1 else 0 end) as correct,
    count(*) as total,
    (sum(case when is_correct then 1 else 0 end)::float / count(*)) as mastery
  from quiz_attempts
  where user_id = match_user_id
  group by topic
  order by mastery asc, total desc;
$$;
