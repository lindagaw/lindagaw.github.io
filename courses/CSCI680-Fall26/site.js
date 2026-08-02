/* CSCI 680 — semester signal.
   One frame per meeting. Bar height = papers assigned that day.
   Gaps are the weeks we don't meet. Diamonds are project deadlines. */

const MEETINGS = [
  { n: 1,  unit: 1, iso: '2026-08-27', date: 'Thu Aug 27', title: 'Course introduction', papers: 1 },
  { n: 2,  unit: 1, iso: '2026-09-01', date: 'Tue Sep 1',  title: 'Can emotion be inferred from behavior?', papers: 1 },
  { n: 3,  unit: 1, iso: '2026-09-03', date: 'Thu Sep 3',  title: 'Corpora I: acted and elicited emotion', papers: 2 },
  { n: 4,  unit: 1, iso: '2026-09-08', date: 'Tue Sep 8',  title: 'Corpora II: naturalistic and in-the-wild data', papers: 2 },
  { n: 5,  unit: 2, iso: '2026-09-10', date: 'Thu Sep 10', title: 'Self-supervised speech representation learning', papers: 1 },
  { n: 6,  unit: 2, iso: '2026-09-15', date: 'Tue Sep 15', title: 'Masked prediction and full-stack pretraining', papers: 2 },
  { n: 7,  unit: 2, iso: '2026-09-17', date: 'Thu Sep 17', title: 'What do these representations actually encode?', papers: 1 },
  { due: 'Project proposal', date: 'Fri Sep 18', href: 'assignments.html#proposal' },
  { n: 8,  unit: 2, iso: '2026-09-22', date: 'Tue Sep 22', title: 'Prompting and parameter-efficient adaptation', papers: 2 },
  { n: 9,  unit: 3, iso: '2026-09-24', date: 'Thu Sep 24', title: 'Audio-language models I', papers: 2 },
  { n: 10, unit: 3, iso: '2026-09-29', date: 'Tue Sep 29', title: 'Audio-language models II', papers: 2 },
  { n: 11, unit: 3, iso: '2026-10-01', date: 'Thu Oct 1',  title: 'Making audio LLMs emotion-aware', papers: 1 },
  { n: 12, unit: 3, iso: '2026-10-06', date: 'Tue Oct 6',  title: 'Structured prompting and low-resource affect', papers: 2 },
  { gap: 'Fall Break, Oct 8–11' },
  { n: 13, unit: 3, iso: '2026-10-13', date: 'Tue Oct 13', title: 'Adapting at test time', papers: 2 },
  { n: 14, unit: 4, iso: '2026-10-15', date: 'Thu Oct 15', title: 'Neural audio codecs', papers: 2 },
  { n: 15, unit: 4, iso: '2026-10-20', date: 'Tue Oct 20', title: 'Codecs that preserve affect', papers: 2 },
  { n: 16, unit: 4, iso: '2026-10-22', date: 'Thu Oct 22', title: 'Zero-shot text-to-speech', papers: 2 },
  { n: 17, unit: 4, iso: '2026-10-27', date: 'Tue Oct 27', title: 'Style and expressivity', papers: 1 },
  { n: 18, unit: 4, iso: '2026-10-29', date: 'Thu Oct 29', title: 'Aligning generative models with preferences', papers: 2 },
  { due: 'Midterm progress report', date: 'Fri Oct 30', href: 'assignments.html#midterm' },
  { gap: 'Election Day, Nov 3' },
  { n: 19, unit: 4, iso: '2026-11-05', date: 'Thu Nov 5',  title: 'Preference optimization for emotional TTS', papers: 2 },
  { n: 20, unit: 5, iso: '2026-11-10', date: 'Tue Nov 10', title: 'Opening the black box: interpretable emotion control', papers: 2 },
  { n: 21, unit: 5, iso: '2026-11-12', date: 'Thu Nov 12', title: 'Domain adaptation', papers: 2 },
  { n: 22, unit: 5, iso: '2026-11-17', date: 'Tue Nov 17', title: 'Distribution shift in deployed affect systems', papers: 2 },
  { n: 23, unit: 5, iso: '2026-11-19', date: 'Thu Nov 19', title: 'Affect sensing in the wild: health deployments', papers: 2 },
  { n: 24, unit: 5, iso: '2026-11-24', date: 'Tue Nov 24', title: 'Beyond speech: multimodal and physiological affect', papers: 2, note: 'online' },
  { gap: 'Thanksgiving Break, Nov 25–29' },
  { n: 25, unit: 6, iso: '2026-12-01', date: 'Tue Dec 1',  title: 'Final project presentations, session I', papers: 0 },
  { n: 26, unit: 6, iso: '2026-12-03', date: 'Thu Dec 3',  title: 'Final project presentations, session II', papers: 0 },
  { due: 'Final report', date: 'Fri Dec 11', href: 'assignments.html#final' }
];

const HEIGHTS = { 0: '32%', 1: '52%', 2: '86%' };

function buildSignal() {
  const track = document.getElementById('track');
  const readout = document.getElementById('readout');
  if (!track) return;

  let i = 0;
  MEETINGS.forEach(item => {
    if (item.gap) {
      const g = document.createElement('span');
      g.className = 'gap';
      g.title = 'No class — ' + item.gap;
      track.appendChild(g);
      return;
    }
    if (item.due) {
      const t = document.createElement('a');
      t.className = 'tick';
      t.href = item.href;
      t.innerHTML = '<span class="sr-only">' + item.due + ' due ' + item.date + '</span>';
      t.addEventListener('mouseenter', () => show({ date: item.date, title: item.due + ' due', meta: 'deadline' }));
      t.addEventListener('focus', () => show({ date: item.date, title: item.due + ' due', meta: 'deadline' }));
      track.appendChild(t);
      return;
    }
    const a = document.createElement('a');
    a.className = 'bar';
    a.href = 'schedule.html#m' + item.n;
    a.dataset.unit = item.unit;
    if (item.papers === 0) a.dataset.kind = 'present';
    a.style.setProperty('--h', HEIGHTS[item.papers]);
    a.style.setProperty('--i', i++);
    const load = item.papers === 0 ? 'student presentations' :
                 item.papers === 1 ? '1 paper' : '2 papers';
    a.innerHTML = '<span class="bar-fill"></span>' +
      '<span class="sr-only">Meeting ' + item.n + ', ' + item.date + ': ' + item.title + '</span>';
    const payload = { date: item.date, title: item.title, meta: 'Meeting ' + item.n + ' · ' + load + (item.note ? ' · ' + item.note : '') };
    a.addEventListener('mouseenter', () => show(payload));
    a.addEventListener('focus', () => show(payload));
    track.appendChild(a);
  });

  function show(d) {
    readout.innerHTML = '<span class="rd-date">' + d.date + '</span><b>' + d.title + '</b><span>' + d.meta + '</span>';
  }

  const first = MEETINGS.find(m => m.n);
  show({ date: first.date, title: first.title, meta: 'Meeting 1 · hover or tab through the strip' });
  track.addEventListener('mouseleave', () =>
    show({ date: first.date, title: first.title, meta: 'Meeting 1 · hover or tab through the strip' }));
}

function nextMeeting() {
  const el = document.getElementById('next-up');
  if (!el) return;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const upcoming = MEETINGS.filter(m => m.iso).find(m => new Date(m.iso + 'T23:59:59') >= today);
  if (!upcoming) { el.textContent = 'The semester is over. Final reports were due Friday, December 11.'; return; }
  const days = Math.round((new Date(upcoming.iso + 'T11:00:00') - today) / 86400000);
  const when = days <= 0 ? 'today' : days === 1 ? 'tomorrow' : 'in ' + days + ' days';
  el.innerHTML = 'Next meeting <b>' + when + '</b> — ' + upcoming.date +
    ', <a href="schedule.html#m' + upcoming.n + '">' + upcoming.title + '</a>.';
}

document.addEventListener('DOMContentLoaded', () => { buildSignal(); nextMeeting(); });
