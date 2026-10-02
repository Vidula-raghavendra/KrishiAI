import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    Sprout, CloudRain, Activity, MapPin, Droplets, Thermometer,
    ChevronRight, CheckCircle, TrendingUp, Globe
} from 'lucide-react';
import { translations } from '../utils/translations';
import '../App.css';

const ResultsPage = ({ language, setLanguage }) => {
    const location = useLocation();
    const navigate = useNavigate();
    const {
        prediction, erosionRisk, longForecast, aiAdvice,
        recommendation, monthlyForecast, formData,
        inputAreaAcres, slopeInfo
    } = location.state || {};

    const t = translations[language] || translations['en'];
    const [showPaymentModal, setShowPaymentModal] = useState(false);

    useEffect(() => {
        if (!location.state) navigate('/app');
    }, [location, navigate]);

    if (!location.state) return null;

    // Weather aggregates (from real Open-Meteo data)
    const totalRainfall = longForecast
        ? longForecast.reduce((s, d) => s + (d.rain || 0), 0).toFixed(1)
        : 0;
    const avgTemp = longForecast && longForecast.length
        ? Math.round(longForecast.reduce((s, d) => s + d.temp, 0) / longForecast.length)
        : '--';
    const avgHumidity = longForecast && longForecast.length
        ? Math.round(longForecast.reduce((s, d) => s + (d.humidity || 70), 0) / longForecast.length)
        : '--';

    // Yield display (model returns tons/ha; user entered acres)
    const yieldPerAcre = prediction?.yield ? (prediction.yield / 2.47105).toFixed(2) : '0.00';
    const totalProd    = prediction?.production ? prediction.production.toFixed(2) : '0.00';

    // Slope display
    const slopeDeg  = slopeInfo?.degrees || (formData?.slope === 'Steep' ? 25 : 10);
    const slopeType = slopeInfo?.slope   || formData?.slope || 'Gentle';

    const handleUnlockPro = () => {
        setShowPaymentModal(true);
        setTimeout(() => {
            setShowPaymentModal(false);
            navigate('/pro-analysis', {
                state: {
                    prediction, erosionRisk, longForecast, aiAdvice,
                    recommendation, formData
                }
            });
        }, 1800);
    };

    return (
        <div className="app-container editorial-theme">
            <div className="grid-lines fixed">
                <div className="line" /><div className="line" />
                <div className="line" /><div className="line" />
            </div>

            <header className="app-header editorial-header">
                <div className="brand" onClick={() => navigate('/')}
                    style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <Sprout size={32} color="#1A3C34" />
                    <h1 style={{ color: '#1A3C34', margin: 0, fontSize: '1.5rem' }}>{t.brand}</h1>
                </div>
                <div className="header-right" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div className="lang-selector" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Globe size={18} color="#1A3C34" />
                        <select value={language} onChange={(e) => setLanguage(e.target.value)}
                            style={{ border: 'none', background: 'transparent', fontSize: '0.9rem',
                                     color: '#1A3C34', cursor: 'pointer', fontWeight: 'bold' }}>
                            <option value="en">English</option>
                            <option value="hi">हिंदी</option>
                            <option value="kha">Khasi</option>
                        </select>
                    </div>
                    <button className="editorial-btn-small" onClick={() => navigate('/app')}>
                        {t.new_analysis || 'New Analysis'}
                    </button>
                </div>
            </header>

            <main className="main-content-dashboard relative-z">
                <div className="dashboard-title">
                    <h2>{t.results_title || 'Analysis Results'}</h2>
                    <p>{t.results_subtitle || 'District:'} {formData?.district}</p>
                </div>

                <motion.div className="results-section"
                    initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>

                    {/* ── Yield Prediction Card ─────────────────────────────── */}
                    {prediction && (
                        <div className="yield-card editorial-card"
                            style={{ marginBottom: '2rem', borderLeft: '4px solid #1A3C34' }}>
                            <div className="yield-header"
                                style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                                <div className="icon-bg"
                                    style={{ background: '#e6f4f1', padding: '10px', borderRadius: '50%' }}>
                                    <TrendingUp size={24} color="#1A3C34" />
                                </div>
                                <div>
                                    <h3 style={{ margin: 0, color: '#1A3C34' }}>
                                        {t.yield_prediction || 'Yield Prediction'}
                                    </h3>
                                    <span style={{ fontSize: '0.85rem', color: '#666' }}>
                                        {formData?.crop} &nbsp;|&nbsp; {inputAreaAcres} acres
                                        {prediction.rainfall_used != null && <> &nbsp;|&nbsp; Rainfall used: {prediction.rainfall_used} mm</>}
                                    </span>
                                </div>
                            </div>

                            <div className="yield-stats"
                                style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1.5rem' }}>
                                <div className="stat-box">
                                    <span className="label"
                                        style={{ display: 'block', fontSize: '0.85rem', color: '#666', marginBottom: '0.25rem' }}>
                                        Yield
                                    </span>
                                    <span className="value"
                                        style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#1A3C34' }}>
                                        {yieldPerAcre}
                                        <span style={{ fontSize: '1rem', fontWeight: 'normal' }}> t/acre</span>
                                    </span>
                                </div>
                                <div className="stat-box">
                                    <span className="label"
                                        style={{ display: 'block', fontSize: '0.85rem', color: '#666', marginBottom: '0.25rem' }}>
                                        Total Production
                                    </span>
                                    <span className="value"
                                        style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#1A3C34' }}>
                                        {totalProd}
                                        <span style={{ fontSize: '1rem', fontWeight: 'normal' }}> tonnes</span>
                                    </span>
                                    {prediction.production_range && (
                                        <span style={{ display: 'block', fontSize: '0.75rem', color: '#888' }}>
                                            likely range {prediction.production_range[0]}–{prediction.production_range[1]} t
                                        </span>
                                    )}
                                </div>
                                {prediction.historical_context && (
                                    <div className="stat-box">
                                        <span className="label"
                                            style={{ display: 'block', fontSize: '0.85rem', color: '#666', marginBottom: '0.25rem' }}>
                                            Historical Avg Yield
                                        </span>
                                        <span className="value"
                                            style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#555' }}>
                                            {prediction.historical_context.avg_historical_yield}
                                            <span style={{ fontSize: '0.9rem', fontWeight: 'normal' }}> t/ha</span>
                                        </span>
                                        <span style={{ fontSize: '0.75rem', color: '#888' }}>
                                            from real crop records
                                        </span>
                                    </div>
                                )}
                            </div>
                            {prediction.confidence_note && (
                                <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#92400e',
                                    background: '#fef3c7', padding: '0.6rem 0.8rem', borderRadius: '8px' }}>
                                    {prediction.confidence_note}
                                </p>
                            )}
                        </div>
                    )}

                    {/* ── Weather + Erosion Row ─────────────────────────────── */}
                    <div className="results-top-row">
                        {erosionRisk && (
                            <div className="erosion-card editorial-card"
                                style={{ background: erosionRisk.color === '#ef4444' ? '#fee2e2' : '#1A3C34', color: 'white' }}>
                                <div className="erosion-header">
                                    <h3>{t.erosion_risk || 'Erosion Risk'}</h3>
                                    <div className="risk-percentage">{erosionRisk.percentage}%</div>
                                    <div className="soil-badge">
                                        <CheckCircle size={14} />
                                        {erosionRisk.risk === 'Low' || erosionRisk.risk === 'Minimal'
                                            ? (t.soil_stable || 'Soil Stable')
                                            : (t.soil_unstable || 'Soil at Risk')}
                                    </div>
                                </div>
                                <div className="recommendation-box">
                                    <h4>{t.recommendation || 'Recommendation'}</h4>
                                    <p>
                                        {erosionRisk.risk === 'Low' || erosionRisk.risk === 'Minimal'
                                            ? (t.cond_optimal || 'Conditions are optimal.')
                                            : (t.cond_risky || 'Implement erosion controls.')}
                                        {' '}
                                        {erosionRisk.risk === 'Low' || erosionRisk.risk === 'Minimal'
                                            ? (t.proceed_standard || 'Proceed with standard practices.')
                                            : (t.proceed_caution || 'Use contour bunding and cover crops.')}
                                    </p>
                                </div>
                            </div>
                        )}

                        <div className="weather-grid-cards">
                            <div className="weather-stat-card editorial-card">
                                <div className="stat-icon-bg"><CloudRain size={20} color="#3b82f6" /></div>
                                <span className="stat-label">{t.rainfall || 'Rainfall'}</span>
                                <span className="stat-value">{totalRainfall} mm</span>
                                <span className="stat-sub">Next 14 Days (live)</span>
                            </div>
                            <div className="weather-stat-card editorial-card">
                                <div className="stat-icon-bg"><MapPin size={20} color="#8b5cf6" /></div>
                                <span className="stat-label">{t.hill_angle || 'Slope'}</span>
                                <span className="stat-value">{slopeDeg}°</span>
                                <span className="stat-sub">{slopeType} terrain</span>
                            </div>
                            <div className="weather-stat-card editorial-card">
                                <div className="stat-icon-bg"><Droplets size={20} color="#0ea5e9" /></div>
                                <span className="stat-label">{t.humidity || 'Humidity'}</span>
                                <span className="stat-value">{avgHumidity}%</span>
                                <span className="stat-sub">14-day avg (live)</span>
                            </div>
                            <div className="weather-stat-card editorial-card">
                                <div className="stat-icon-bg"><Thermometer size={20} color="#f97316" /></div>
                                <span className="stat-label">{t.temperature || 'Temp'}</span>
                                <span className="stat-value">{avgTemp}°C</span>
                                <span className="stat-sub">14-day avg (live)</span>
                            </div>
                        </div>
                    </div>

                    {/* ── Advisory + Monthly Forecast + Pro ────────────────── */}
                    <div className="results-bottom-row">
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                            {/* Advisory */}
                            {aiAdvice && (
                                <div className="advisory-card editorial-card">
                                    <div className="advisory-header">
                                        <div className="advisory-icon-bg">
                                            <Sprout size={24} color="#1A3C34" />
                                        </div>
                                        <h3>{t.ai_advisory || 'Advisory'}</h3>
                                    </div>

                                    <div className="advisory-item">
                                        <div className="advisory-icon"><Sprout size={18} color="#1A3C34" /></div>
                                        <div>
                                            <span className="advisory-label">Best Crop Recommendation</span>
                                            <div className="advisory-value">
                                                {recommendation?.best_crop
                                                    ? `${recommendation.best_crop.crop} — ` +
                                                      `${(recommendation.best_crop.yield / 2.47105).toFixed(2)} t/acre predicted yield`
                                                    : formData?.crop}
                                            </div>
                                        </div>
                                    </div>

                                    {recommendation?.recommendations?.length > 1 && (
                                        <div className="advisory-item">
                                            <div className="advisory-icon"><Activity size={18} color="#8b5cf6" /></div>
                                            <div>
                                                <span className="advisory-label">Top 5 Alternative Crops</span>
                                                <div className="advisory-value" style={{ fontSize: '0.85rem', color: '#444' }}>
                                                    {recommendation.recommendations.map(r =>
                                                        `${r.crop} (${(r.yield / 2.47105).toFixed(2)} t/acre)`
                                                    ).join(' · ')}
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {aiAdvice.alerts?.length > 0 && (
                                        <div className="advisory-item">
                                            <div className="advisory-icon"><CloudRain size={18} color="#ef4444" /></div>
                                            <div>
                                                <span className="advisory-label">Weather Alerts (live forecast)</span>
                                                {aiAdvice.alerts.slice(0, 3).map((alert, idx) => (
                                                    <div key={idx} className="advisory-value"
                                                        style={{ color: '#ef4444', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                                                        ⚠ {alert}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {aiAdvice.crop_advice && (
                                        <div className="advisory-item">
                                            <div className="advisory-icon"><Activity size={18} color="#1A3C34" /></div>
                                            <div>
                                                <span className="advisory-label">About {formData?.crop}</span>
                                                <div className="advisory-value"
                                                    style={{ fontSize: '0.85rem', color: '#444', lineHeight: '1.5' }}>
                                                    {aiAdvice.crop_advice.slice(0, 300)}
                                                    {aiAdvice.crop_advice.length > 300 ? '...' : ''}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Monthly Rainfall Forecast (from real CSV) */}
                            {monthlyForecast && monthlyForecast.length > 0 && (
                                <div className="editorial-card" style={{ padding: '1.5rem' }}>
                                    <h3 style={{ margin: '0 0 1rem 0', color: '#1A3C34', fontSize: '1rem' }}>
                                        6-Month Rainfall Forecast — {formData?.district}
                                    </h3>
                                    <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                                        {monthlyForecast.map((m, i) => (
                                            <div key={i} style={{
                                                flex: '1 1 80px', background: '#f0f7f4',
                                                borderRadius: '8px', padding: '0.75rem',
                                                textAlign: 'center', minWidth: '70px'
                                            }}>
                                                <div style={{ fontWeight: 'bold', color: '#1A3C34', fontSize: '0.9rem' }}>
                                                    {m.month}
                                                </div>
                                                <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#0ea5e9' }}>
                                                    {m.rainfall} mm
                                                </div>
                                                <div style={{
                                                    fontSize: '0.7rem', marginTop: '0.25rem',
                                                    color: m.erosion_risk === 'High' ? '#ef4444'
                                                         : m.erosion_risk === 'Moderate' ? '#f59e0b' : '#10b981'
                                                }}>
                                                    {m.erosion_risk} risk
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Pro Upgrade Card */}
                        <div className="pro-card editorial-card">
                            <div className="pro-badge">👑 {t.premium_access || 'Premium'}</div>
                            <h3>{t.upgrade_pro || 'Upgrade to Pro'}</h3>
                            <p>{t.unlock_desc || 'Get deeper insights powered by your real data.'}</p>

                            <div className="pro-features">
                                <div className="pro-feature">
                                    <div className="pf-icon">📈</div>
                                    <span>Market price estimates from crop yield + MSP data</span>
                                </div>
                                <div className="pro-feature">
                                    <div className="pf-icon">🐞</div>
                                    <span>Pest risk forecast from live weather patterns</span>
                                </div>
                                <div className="pro-feature">
                                    <div className="pf-icon">🌱</div>
                                    <span>Soil nutrient recommendations per crop</span>
                                </div>
                                <div className="pro-feature">
                                    <div className="pf-icon">📅</div>
                                    <span>Crop calendar from your sowing date</span>
                                </div>
                            </div>

                            <div className="price-tag">
                                <div className="price">₹100 <span className="period">/ season</span></div>
                            </div>

                            <button className="pay-btn" onClick={handleUnlockPro}>
                                {showPaymentModal ? 'Processing...' : (t.pay_btn || 'Unlock Pro')}
                                <ChevronRight size={16} />
                            </button>
                        </div>
                    </div>
                </motion.div>
            </main>

            {showPaymentModal && (
                <div className="payment-overlay">
                    <div className="payment-spinner"></div>
                    <p>Processing Secure Payment...</p>
                </div>
            )}
        </div>
    );
};

export default ResultsPage;
