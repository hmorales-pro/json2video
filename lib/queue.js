/**
 * Queue de jobs in-process, ultra-simple.
 *
 * Pourquoi pas BullMQ/Redis ? Pour rester sans dépendance d'infra.
 * Limitations à connaître :
 *   - Tout vit en RAM : un crash du process perd les jobs en cours.
 *   - Mono-instance : ne se réplique pas sur plusieurs containers.
 *
 * Pour passer en prod multi-instance, remplacer cette implémentation
 * par BullMQ en gardant la même API (enqueue/get/list).
 */
const { v4: uuidv4 } = require("uuid");
const { EventEmitter } = require("events");

const JOB_TTL_MS = 24 * 60 * 60 * 1000; // 24 h après "completed"/"failed"
const GC_INTERVAL_MS = 10 * 60 * 1000;  // GC toutes les 10 min

class JobQueue extends EventEmitter {
  constructor({ concurrency = 1 } = {}) {
    super();
    this.concurrency = Math.max(1, concurrency | 0);
    this.jobs = new Map();
    this.pending = [];
    this.running = 0;
    setInterval(() => this._gc(), GC_INTERVAL_MS).unref();
  }

  /**
   * @param {Function} task  Fonction async () => result.
   * @param {object} meta    Métadonnées attachées au job (pour affichage).
   * @returns {string} jobId
   */
  enqueue(task, meta = {}) {
    const id = uuidv4();
    const job = {
      id,
      status: "queued",
      meta,
      created_at: new Date().toISOString(),
      started_at: null,
      finished_at: null,
      result: null,
      error: null,
      _task: task,
    };
    this.jobs.set(id, job);
    this.pending.push(id);
    this._drain();
    return id;
  }

  get(id) {
    const j = this.jobs.get(id);
    if (!j) return null;
    // Renvoie une copie sans la closure _task.
    const { _task, ...safe } = j;
    return safe;
  }

  list({ status } = {}) {
    return [...this.jobs.values()]
      .filter((j) => !status || j.status === status)
      .map(({ _task, ...rest }) => rest)
      .sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  }

  _drain() {
    while (this.running < this.concurrency && this.pending.length) {
      const id = this.pending.shift();
      const job = this.jobs.get(id);
      if (!job) continue;
      this.running++;
      job.status = "processing";
      job.started_at = new Date().toISOString();
      this.emit("started", job.id);

      Promise.resolve()
        .then(() => job._task(job.id))  // passe job.id au task pour les webhooks
        .then((result) => {
          job.status = "completed";
          job.result = result;
          job.finished_at = new Date().toISOString();
          this.emit("completed", job.id);
        })
        .catch((err) => {
          job.status = "failed";
          job.error = err?.message || String(err);
          job.finished_at = new Date().toISOString();
          this.emit("failed", job.id);
        })
        .finally(() => {
          this.running--;
          this._drain();
        });
    }
  }

  _gc() {
    const now = Date.now();
    for (const [id, job] of this.jobs.entries()) {
      if (job.status === "completed" || job.status === "failed") {
        const finished = Date.parse(job.finished_at || job.created_at);
        if (now - finished > JOB_TTL_MS) this.jobs.delete(id);
      }
    }
  }
}

// Singleton process-wide
const defaultQueue = new JobQueue({
  concurrency: parseInt(process.env.RENDER_CONCURRENCY || "1", 10),
});

module.exports = { JobQueue, defaultQueue };
