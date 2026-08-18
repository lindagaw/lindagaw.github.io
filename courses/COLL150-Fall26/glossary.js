/* COLL 150 — the doubled glossary.
   Six words this course uses twice: once as a mechanism,
   once as a story. Each one points at the unit where we
   argue about it. */

const GLOSSARY = [
  {
    word: 'learn',
    machine: 'Adjust a few billion parameters to reduce a single number that says how wrong you currently are. Repeat. <b>That is the whole thing</b> — that, and a great deal of electricity.',
    story: 'The creature learns language by eavesdropping through a wall on a family that will reject him on sight. He teaches himself to read from <b>three books he found in a bag</b>.',
    unitLabel: 'Units 2 & 5 · Frankenstein · Loss functions',
    href: 'schedule.html#u5'
  },
  {
    word: 'collapse',
    machine: 'A generator discovers that a small number of outputs reliably satisfy its critic, so it produces those, and only those, forever. <b>It has not broken. It is succeeding</b> — narrowly and perfectly.',
    story: 'What happens to anyone rewarded for the answer that always pleases. The diversity of the world was never part of the objective, so <b>the diversity of the world quietly disappears</b>.',
    unitLabel: 'Unit 6 · Mode collapse',
    href: 'schedule.html#u6'
  },
  {
    word: 'want',
    machine: 'An agent, an environment, a reward. The agent does not know what it is doing and does not need to. It flails; <b>the flails that pay are reinforced</b>; a policy exists that no one wrote.',
    story: 'Faustus sells his soul for the secrets of the cosmos and then, catastrophically, cannot think of anything to do with them. <b>He had everything, and he wanted parlor tricks</b>.',
    unitLabel: 'Units 7 & 8 · Faustus · Reinforcement learning',
    href: 'schedule.html#u8'
  },
  {
    word: 'forget',
    machine: 'Fine-tune a model on a new task and it can lose the old one — not gradually but all at once. The field named this <b>catastrophic forgetting</b>, and we are going to think about why an engineer reached for that word.',
    story: 'Funes remembers everything and is therefore incapable of thought. <b>To think is to forget</b>, to generalize, to let the particular go. What is the right amount to lose?',
    unitLabel: 'Units 9 & 12 · Borges · Catastrophic forgetting',
    href: 'schedule.html#u12'
  },
  {
    word: 'pass',
    machine: 'The benchmark, the eval, the leaderboard. A score that stands in for a capacity, on a test somebody designed, <b>which everything downstream then inherits</b>.',
    story: 'Turing opens with a party game about gender: a man pretends to be a woman, and the interrogator guesses. <b>Passing is the frame from the very first page</b>.',
    unitLabel: 'Unit 10 · The imitation game',
    href: 'schedule.html#u10'
  },
  {
    word: 'work',
    machine: 'Preference data, written by specific people paid specific amounts, who rank and rate and moderate so a model can learn what not to reproduce. <b>RLHF has a payroll</b>.',
    story: '<i>Robota</i>: the drudgery owed by a serf. The word arrives in 1920 in a Czech play and we spent a century politely forgetting it. <b>There is still a person in the box</b>.',
    unitLabel: 'Unit 11 · Robota, labor, and the ghost in the machine',
    href: 'schedule.html#u11'
  }
];

function initGlossary() {
  const list = document.getElementById('words');
  const mBody = document.getElementById('reg-machine-body');
  const sBody = document.getElementById('reg-story-body');
  const src = document.getElementById('reg-source');
  if (!list) return;

  GLOSSARY.forEach((entry, idx) => {
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.className = 'word';
    b.type = 'button';
    b.textContent = entry.word;
    b.setAttribute('aria-pressed', idx === 0 ? 'true' : 'false');
    b.addEventListener('click', () => select(idx));
    li.appendChild(b);
    list.appendChild(li);
  });

  function select(idx) {
    const e = GLOSSARY[idx];
    list.querySelectorAll('.word').forEach((b, i) =>
      b.setAttribute('aria-pressed', i === idx ? 'true' : 'false'));
    mBody.innerHTML = e.machine;
    sBody.innerHTML = e.story;
    src.innerHTML = '<a class="src" href="' + e.href + '">' + e.unitLabel + ' →</a>';
    [mBody, sBody].forEach(el => {
      el.classList.remove('fade-in');
      void el.offsetWidth;
      el.classList.add('fade-in');
    });
  }

  select(0);
}

document.addEventListener('DOMContentLoaded', initGlossary);
