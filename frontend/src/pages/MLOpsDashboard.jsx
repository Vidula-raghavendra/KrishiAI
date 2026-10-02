import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, RefreshCw, Cpu, AlertTriangle, BarChart2, Activity } from 'lucide-react';
import './MLOpsDashboard.css';

const API = 'http://127.0.0.1:5000';

const STATUS_COLOR = { green: '#22c55e', amber: '#f59e0b', red: '#ef4444' };

export default function MLOpsDashboard({ language }) {
    const navigate = useNavigate();

    const [modelInfo,   setModelInfo]   = useState(null);
    const [drift,       setDrift]       = useState(null);
    const [predStats,   setPredStats]   = useState(null);
    const [retrainResult, setRetrainResult] = useState(null);

    const [loadingInfo,    setLoadingInfo]    = useState(true);
    const [loadingDrift,   setLoadingDrift]   = useState(true);
    const [loadingStats,   setLoadingStats]   = useState(true);
    const [retraining,     setRetraining]     = useState(false);

    const fetchAll = useCallback(() => {
        setLoadingInfo(true);
        setLoadingDrift(true);
        setLoadingStats(true);

        fetch(`${API}/model-info`)
            .then(r => r.json()).then(setModelInfo).finally(() => setLoadingInfo(false));

        fetch(`${API}/drift-report`)
            .then(r => r.json()).then(setDrift).finally(() => setLoadingDrift(false));

        fetch(`${API}/prediction-stats`)
            .then(r => r.json()).then(setPredStats).finally(() => setLoadingStats(false));
    }, []);

    useEffect(() => { fetchAll(); }, [fetchAll]);

    const handleRetrain = async () => {
        setRetraining(true);
        setRetrainResult(null);
        try {
            const res  = await fetch(`${API}/retrain`, { method: 'POST' });
            const data = await res.json();
            setRetrainResult(data);
            if (data.status === 'upgraded' || data.status === 'refreshed') fetchAll();
        } catch (e) {
            setRetrainResult({ error: e.message });
        } finally {
            setRetraining(false);
        }
    };

    const r2Pct = modelInfo?.r2 != null ? Math.round(modelInfo.r2 * 100) : null;

    // Top-3 districts by prediction volume
    const topDistricts = predStats?.by_district
        ? Object.entries(predStats.by_district)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 6)
        : [];
    const maxCount = topDistricts[0]?.[1] || 1;

    return (
        <div className="mlops-container">
            {/* Grid lines (same editorial theme) */}
            <div className="grid-lines">
                {[...Array(4)].map((_, i) => <div key={i} className="line" />)}
            </div>

            {/* Header */}
            <nav className="mlops-nav">
                <button className="back-btn" onClick={() => navigate('/')}>
                    <ArrowLeft size={16} /> Home
                </button>
                <span className="mlops-brand">DATANYX <span className="brand-sub">/ MLOps</span></span>
                <button className="refresh-btn" onClick={fetchAll}>
                    <RefreshCw size={14} /> Refresh
                </button>
            </nav>

            <div className="mlops-title-section">
                <h1 className="mlops-headline">Model <span className="italic-accent">Operations</span></h1>
                <p className="mlops-sub">Live pipeline health for the Meghalaya yield prediction model</p>
            </div>

            <div className="mlops-grid">

                {/* ── Card 1: Model Health ─────────────────────────────────── */}
                <div className="mlops-card model-health-card">
                    <div className="card-header">
                        <Cpu size={18} /> <span>Model Health</span>
                    </div>

                    {loadingInfo ? <div className="skeleton-loader" /> : modelInfo ? (
                        <>
                            <div className="metric-row">
                                <span className="metric-label">Model</span>
                                <span className="metric-value">{modelInfo.model_name || 'agriculture-yield-model'}</span>
                            </div>
                            <div className="metric-row">
                                <span className="metric-label">Version</span>
                                <span className="metric-value badge">v{modelInfo.version || 1}</span>
                            </div>
                            <div className="metric-row">
                                <span className="metric-label">Trained</span>
                                <span className="metric-value">
                                    {modelInfo.training_date !== 'unknown'
                                        ? new Date(modelInfo.training_date).toLocaleDateString()
                                        : 'Unknown'}
                                </span>
                            </div>
                            <div className="metric-row">
                                <span className="metric-label">Training Samples</span>
                                <span className="metric-value">{modelInfo.n_samples?.toLocaleString?.() ?? modelInfo.n_samples}</span>
                            </div>

                            {/* R² gauge */}
                            {r2Pct != null && (
                                <div className="gauge-section">
                                    <div className="gauge-label">
                                        R² Score <span className="gauge-value">{modelInfo.r2}</span>
                                    </div>
                                    <div className="gauge-bar">
                                        <div
                                            className="gauge-fill"
                                            style={{
                                                width: `${r2Pct}%`,
                                                background: r2Pct > 85 ? '#22c55e' : r2Pct > 70 ? '#f59e0b' : '#ef4444'
                                            }}
                                        />
                                    </div>
                                </div>
                            )}

                            <div className="metrics-row">
                                <div className="mini-metric">
                                    <span className="mini-label">MAE</span>
                                    <span className="mini-val">{modelInfo.mae ?? '—'}</span>
                                </div>
                                <div className="mini-metric">
                                    <span className="mini-label">RMSE</span>
                                    <span className="mini-val">{modelInfo.rmse ?? '—'}</span>
                                </div>
                            </div>

                            {modelInfo.champion && (
                                <p style={{ fontSize: '0.8rem', color: '#555', margin: '0.75rem 0 0.25rem' }}>
                                    Deployed: <b>{modelInfo.champion}</b> — {modelInfo.champion_description}.
                                    Errors in t/ha on a time-based backtest.
                                </p>
                            )}
                            {modelInfo.leaderboard?.length > 0 && (
                                <table style={{ width: '100%', fontSize: '0.78rem', borderCollapse: 'collapse', marginTop: '0.5rem' }}>
                                    <thead>
                                        <tr style={{ textAlign: 'left', color: '#666' }}>
                                            <th>Candidate</th><th>MAE</th><th>R²</th><th>Median % err</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {modelInfo.leaderboard.map(r => (
                                            <tr key={r.model} style={{ fontWeight: r.model === modelInfo.champion ? 700 : 400 }}>
                                                <td>{r.model}</td>
                                                <td>{r.mae?.toFixed(3)}</td>
                                                <td>{r.r2_yield?.toFixed(3)}</td>
                                                <td>{r.mdape_pct?.toFixed(1)}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            )}
                        </>
                    ) : <p className="no-data">Run <code>python -m ml.pipeline train</code> to populate model info.</p>}
                </div>

                {/* ── Card 2: Drift Detection ──────────────────────────────── */}
                <div className="mlops-card drift-card">
                    <div className="card-header">
                        <AlertTriangle size={18} /> <span>Data Drift Detection</span>
                    </div>

                    {loadingDrift ? <div className="skeleton-loader" /> : drift ? (
                        <>
                            <div className={`drift-status-badge ${drift.status === 'ok' ? 'ok' : drift.status === 'insufficient_data' ? 'warn' : 'alert'}`}>
                                {drift.status === 'ok' ? 'No Drift Detected'
                                    : drift.status === 'insufficient_data' ? 'Insufficient Data'
                                    : 'Drift Detected'}
                            </div>

                            <div className="drift-meta">
                                <span>Engine: <b>{drift.engine || 'N/A'}</b></span>
                                <span>Ref: <b>{drift.n_reference?.toLocaleString?.() ?? '—'}</b> rows</span>
                                <span>Current: <b>{drift.n_current}</b> predictions</span>
                            </div>

                            {drift.message && <p className="drift-message">{drift.message}</p>}

                            {drift.features && Object.keys(drift.features).length > 0 && (
                                <div className="feature-drift-list">
                                    {Object.entries(drift.features).map(([feat, info]) => (
                                        <div key={feat} className="feature-drift-row">
                                            <span className="feat-name">{feat}</span>
                                            <span className="feat-pval">p={info.p_value}</span>
                                            <span
                                                className="traffic-light"
                                                style={{ background: STATUS_COLOR[info.status] || '#ccc' }}
                                                title={info.drifted ? 'Drifted' : 'Stable'}
                                            />
                                        </div>
                                    ))}
                                </div>
                            )}

                            <p className="drift-ts">
                                Last checked: {drift.timestamp ? new Date(drift.timestamp).toLocaleTimeString() : '—'}
                            </p>
                        </>
                    ) : <p className="no-data">No drift data available.</p>}
                </div>

                {/* ── Card 3: Prediction Volume ────────────────────────────── */}
                <div className="mlops-card stats-card">
                    <div className="card-header">
                        <BarChart2 size={18} /> <span>Prediction Volume</span>
                    </div>

                    {loadingStats ? <div className="skeleton-loader" /> : predStats ? (
                        <>
                            <div className="stat-summary">
                                <div className="stat-pill">
                                    <span className="stat-num">{predStats.count}</span>
                                    <span className="stat-lbl">Total (last 100)</span>
                                </div>
                                <div className="stat-pill">
                                    <span className="stat-num">{predStats.avg_area_ha ?? '—'}</span>
                                    <span className="stat-lbl">Avg Area (ha)</span>
                                </div>
                            </div>

                            {topDistricts.length > 0 ? (
                                <div className="bar-chart">
                                    {topDistricts.map(([district, count]) => (
                                        <div key={district} className="bar-row">
                                            <span className="bar-label">{district.split(' ').slice(0, 2).join(' ')}</span>
                                            <div className="bar-track">
                                                <div
                                                    className="bar-fill"
                                                    style={{ width: `${(count / maxCount) * 100}%` }}
                                                />
                                            </div>
                                            <span className="bar-count">{count}</span>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p className="no-data">No predictions logged yet. Make some predictions from the main tool.</p>
                            )}

                            {predStats.by_season && Object.keys(predStats.by_season).length > 0 && (
                                <div className="season-tags">
                                    {Object.entries(predStats.by_season).map(([s, c]) => (
                                        <span key={s} className="season-tag">{s}: {c}</span>
                                    ))}
                                </div>
                            )}
                        </>
                    ) : <p className="no-data">No prediction data yet.</p>}
                </div>

                {/* ── Card 4: Retrain ──────────────────────────────────────── */}
                <div className="mlops-card retrain-card">
                    <div className="card-header">
                        <Activity size={18} /> <span>Retraining Pipeline</span>
                    </div>

                    <p className="retrain-desc">
                        Re-scores every candidate on the same time-based backtest, using the base data plus any
                        new harvest records in <code>new_observations.csv</code>. A challenger replaces the deployed
                        model only if it cuts MAE by at least 2%.
                    </p>

                    <div className="pipeline-steps">
                        {['Load Data', 'Backtest Candidates', 'Re-score Deployed', 'Compare', 'Promote / Refit / Keep'].map((step, i) => (
                            <div key={step} className="pipeline-step">
                                <span className="step-num">{i + 1}</span>
                                <span className="step-name">{step}</span>
                            </div>
                        ))}
                    </div>

                    <button
                        className={`retrain-btn ${retraining ? 'loading' : ''}`}
                        onClick={handleRetrain}
                        disabled={retraining}
                    >
                        {retraining ? (
                            <><RefreshCw size={14} className="spin" /> Retraining…</>
                        ) : 'Run Retraining Pipeline'}
                    </button>

                    {retrainResult && (
                        <div className={`retrain-result ${retrainResult.error ? 'error' : retrainResult.status === 'upgraded' ? 'success' : 'neutral'}`}>
                            {retrainResult.error ? (
                                <span>Error: {retrainResult.error}</span>
                            ) : (
                                <>
                                    <strong>{retrainResult.status === 'upgraded' ? 'Model Upgraded' : retrainResult.status === 'refreshed' ? 'Refit on New Data' : 'No Change'}</strong>
                                    <span>New R²: {retrainResult.new_r2} | Old R²: {retrainResult.old_r2}</span>
                                    <span className="retrain-reason">{retrainResult.reason}</span>
                                </>
                            )}
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
}
