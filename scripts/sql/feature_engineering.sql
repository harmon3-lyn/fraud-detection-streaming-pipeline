/*
-- SQL FEATURE ENGINEERING --

FEATURE LIST:
-----------------------------------------------------------------------
KEYS: trans_num, cc_num, merchant, is_fraud
CALENDAR: TransactionHour, DayOfWeek, IsWeekend, IsNightTxn
AMOUNT: AmountZScore, AmountPercentileRank, IsHighAmount, AmtToMaxRatio, AvgTxnAmtLast24h
VELOCITY: TxnCount_1h/6h/24h/7d, TxnAmount_1h/24h/7d, AvgTxnAmtLast24h
SEQUENTIAL: TimeSinceLastTxnSec, IsFirstTxn, IsFirstTimeAtMerchant, MerchantVisitRank, LatDeltaFromLastTxn, SuspiciousGeoSpeed
LIFECYCLE: DaysSinceAccountCreation, DaysSinceFirstTxn, DaysSincePasswordChange
DIVERSITY: TotalDistinctMerchants, TotalDistinctCategories
IDENTITY: AccountAgeDays, IdentityConfidenceScore, AddressConsistencyScore, PreviousFraudFlag, IsDisposableEmailDomain
EMAIL: EmailRiskScore, IsDisposableEmail, EmailAgeDays, DomainReputationScore
PHONE: PhoneRiskScore, IsVOIP, PhoneAgeDays, CarrierType
IP/NETWORK: IPRiskScore, IsVPN, IsProxy, IsTor, IPCountryMismatch, IP_Country
ADDRESS: AddressRiskScore, IsCommercialAddress, GeoConsistencyScore
BUREAU: SyntheticIdentityFlag, SSNMatchScore, IdentityVerificationStatus, CreditScoreOrdinal
DEVICE: IsNewDevice, BotLikelihoodScore, CheckoutTimeSec, SessionDurationSec, PagesViewed, InteractionComplexityScore, DeviceTrustScore, NumAccountsOnDevice, DeviceChangeFrequency, DeviceBlacklisted
MERCHANT: MerchantRiskScore, MerchantFraudRate, IsHighRiskCategory, MerchantCategory, MerchantCountry
ACCOUNT: ChargebackCount, FraudFlagHistory, ChargebackRatio, HistoricalVelocityScore
LOGIN: LoginFailureRate, TotalLoginAttempts
RULES: Rule_HighVelocity, Rule_CountryMismatch, Rule_NewDeviceHighAmount, Rule_DisposableEmail, Rule_VPNorTor, Rule_SyntheticIdentity, Rule_HighChargebacks, Rule_BotLikely, Rule_NewAccount, Rule_VOIPPhone, RiskSignalCount
-----------------------------------------------------------------------
*/




-- STEP 0: Generate single transaction view (train + test)
IF OBJECT_ID('dbo.vw_all_transactions', 'V') IS NOT NULL
    DROP VIEW dbo.vw_all_transactions;
GO

CREATE VIEW dbo.vw_all_transactions AS
    SELECT * FROM dbo.fraud_train
    UNION ALL
    SELECT * FROM dbo.fraud_test;
GO




-- STEP 1: Temp table base
IF OBJECT_ID('tempdb..#base') IS NOT NULL DROP TABLE #base;

SELECT
    CAST(trans_num AS VARCHAR(64)) AS trans_num,
    CAST(cc_num AS BIGINT) AS cc_num,
    CAST(merchant AS VARCHAR(128)) AS merchant,
    CAST(category AS VARCHAR(64)) AS category,
    CAST(amt AS FLOAT) AS amt,
    CAST(unix_time AS BIGINT) AS unix_time,
    CAST(trans_date_trans_time AS DATETIME2) AS trans_dt,
    CAST(lat AS FLOAT) AS cust_lat,
    CAST([long] AS FLOAT) AS cust_long,
    CAST(merch_lat AS FLOAT) AS merch_lat,
    CAST(merch_long AS FLOAT) AS merch_long,
    CAST(is_fraud AS TINYINT) AS is_fraud,
    -- Calendar features
    CAST(DATEPART(HOUR, trans_date_trans_time) AS TINYINT) AS TransactionHour,
    CAST(DATEPART(WEEKDAY, trans_date_trans_time) AS TINYINT) AS DayOfWeek,
    CASE WHEN DATEPART(WEEKDAY, trans_date_trans_time) IN (1,7)
         THEN 1 ELSE 0 END AS IsWeekend,
    CASE WHEN DATEPART(HOUR, trans_date_trans_time) BETWEEN 0 AND 5
         THEN 1 ELSE 0 END AS IsNightTxn
INTO #base
FROM dbo.vw_all_transactions;

-- Composite index for velocity self-joins: (cc_num, unix_time) + amt INCLUDE
CREATE CLUSTERED INDEX CIX_base_cc_time ON #base (cc_num, unix_time);
GO




-- STEP 2: Per-customer stats - TRAIN PERIOD ONLY (2019)
/*
- Using only trans_dt before 2020
- Customers with no 2019 rows fall back to all-time stats
*/
IF OBJECT_ID('tempdb..#cust_stats') IS NOT NULL DROP TABLE #cust_stats;

-- Train-period aggregates
SELECT
    cc_num,
    AVG(amt) AS cust_avg_amt,
    CASE WHEN STDEV(amt) = 0 OR STDEV(amt) IS NULL
         THEN 1.0 ELSE STDEV(amt) END AS cust_std_amt,
    MAX(amt) AS cust_max_amt,
    MIN(CAST(trans_dt AS DATE)) AS cust_first_txn_date,
    COUNT(DISTINCT merchant) AS TotalDistinctMerchants,
    COUNT(DISTINCT category) AS TotalDistinctCategories
INTO #cust_stats
FROM #base
WHERE trans_dt < '2020-01-01'
GROUP BY cc_num;

-- Fall back: customers that only appear in test period (no 2019 rows)
INSERT INTO #cust_stats
SELECT
    b.cc_num,
    AVG(b.amt),
    CASE WHEN STDEV(b.amt) = 0 OR STDEV(b.amt) IS NULL THEN 1.0 ELSE STDEV(b.amt) END,
    MAX(b.amt),
    MIN(CAST(b.trans_dt AS DATE)),
    COUNT(DISTINCT b.merchant),
    COUNT(DISTINCT b.category)
FROM #base b
WHERE NOT EXISTS (SELECT 1 FROM #cust_stats cs WHERE cs.cc_num = b.cc_num)
GROUP BY b.cc_num;

CREATE UNIQUE CLUSTERED INDEX CIX_cs ON #cust_stats (cc_num);
GO




-- STEP 3: Sequential features
IF OBJECT_ID('tempdb..#sequential') IS NOT NULL DROP TABLE #sequential;

SELECT
    trans_num,
    unix_time - LAG(unix_time) OVER (
        PARTITION BY cc_num ORDER BY unix_time) AS TimeSinceLastTxnSec,
    ROW_NUMBER() OVER (
        PARTITION BY cc_num, merchant ORDER BY unix_time) AS MerchantVisitRank,
    ABS(merch_lat - LAG(merch_lat) OVER (
        PARTITION BY cc_num ORDER BY unix_time)) AS LatDeltaFromLastTxn,
    ABS(merch_long - LAG(merch_long) OVER (
        PARTITION BY cc_num ORDER BY unix_time)) AS LongDeltaFromLastTxn,
    CASE
        WHEN (
            ABS(merch_lat - LAG(merch_lat) OVER (PARTITION BY cc_num ORDER BY unix_time))
          + ABS(merch_long - LAG(merch_long) OVER (PARTITION BY cc_num ORDER BY unix_time))
        ) > 5
        AND (unix_time - LAG(unix_time) OVER (PARTITION BY cc_num ORDER BY unix_time)) < 3600
        THEN 1 ELSE 0
    END AS SuspiciousGeoSpeed
INTO #sequential
FROM #base;

CREATE UNIQUE CLUSTERED INDEX CIX_seq ON #sequential (trans_num);
GO




-- STEP 4: Amount percentile rank per customer
IF OBJECT_ID('tempdb..#amt_rank') IS NOT NULL DROP TABLE #amt_rank;

SELECT
    trans_num,
    ROUND(PERCENT_RANK() OVER (PARTITION BY cc_num ORDER BY amt), 4) AS AmountPercentileRank
INTO #amt_rank
FROM #base;

CREATE UNIQUE CLUSTERED INDEX CIX_ar ON #amt_rank (trans_num);
GO




-- STEP 5: Velocity windows via indexed correlated subqueries
/*
- Window sizes: 1h, 6h, 24h, 7d
- Exclude current txn so counts represent prior activity
*/
IF OBJECT_ID('tempdb..#velocity') IS NOT NULL DROP TABLE #velocity;

SELECT
    b.trans_num,

    -- Transaction counts
    (SELECT COUNT(*) FROM #base x WHERE x.cc_num = b.cc_num AND x.unix_time > b.unix_time - 3600 AND x.unix_time < b.unix_time) AS TxnCount_1h,
    (SELECT COUNT(*) FROM #base x WHERE x.cc_num = b.cc_num AND x.unix_time > b.unix_time - 21600 AND x.unix_time < b.unix_time) AS TxnCount_6h,
    (SELECT COUNT(*) FROM #base x WHERE x.cc_num = b.cc_num AND x.unix_time > b.unix_time - 86400 AND x.unix_time < b.unix_time) AS TxnCount_24h,
    (SELECT COUNT(*) FROM #base x WHERE x.cc_num = b.cc_num AND x.unix_time > b.unix_time - 604800 AND x.unix_time < b.unix_time) AS TxnCount_7d,

    -- Spend amounts
    (SELECT COALESCE(SUM(amt), 0) FROM #base x WHERE x.cc_num = b.cc_num AND x.unix_time > b.unix_time - 3600 AND x.unix_time < b.unix_time) AS TxnAmount_1h,
    (SELECT COALESCE(SUM(amt), 0) FROM #base x WHERE x.cc_num = b.cc_num AND x.unix_time > b.unix_time - 86400 AND x.unix_time < b.unix_time) AS TxnAmount_24h,
    (SELECT COALESCE(SUM(amt), 0) FROM #base x WHERE x.cc_num = b.cc_num AND x.unix_time > b.unix_time - 604800 AND x.unix_time < b.unix_time) AS TxnAmount_7d

INTO #velocity
FROM #base b;

CREATE UNIQUE CLUSTERED INDEX CIX_vel ON #velocity (trans_num);
GO




-- STEP 6: Login aggregates per customer
IF OBJECT_ID('tempdb..#login_agg') IS NOT NULL DROP TABLE #login_agg;

SELECT
    cc_num,
    COUNT(*) AS TotalLoginAttempts,
    CAST(SUM(CASE WHEN Success = 0 THEN 1 ELSE 0 END) AS FLOAT)
        / NULLIF(COUNT(*), 0) AS LoginFailureRate
INTO #login_agg
FROM dbo.login_events
GROUP BY cc_num;

CREATE UNIQUE CLUSTERED INDEX CIX_la ON #login_agg (cc_num);
GO




-- STEP 7: Last password change per customer
IF OBJECT_ID('tempdb..#pwd_change') IS NOT NULL DROP TABLE #pwd_change;

SELECT cc_num, MAX(ChangedAt) AS LastChangeDate
INTO #pwd_change
FROM dbo.password_change_log
GROUP BY cc_num;

CREATE UNIQUE CLUSTERED INDEX CIX_pc ON #pwd_change (cc_num);
GO


-- STEP 7b: Per-merchant fraud rate (from train period only)
IF OBJECT_ID('tempdb..#merch_fraud_rate') IS NOT NULL DROP TABLE #merch_fraud_rate;

SELECT
    merchant,
    CAST(SUM(CAST(is_fraud AS FLOAT)) / NULLIF(COUNT(*), 0) AS FLOAT) AS train_fraud_rate
INTO #merch_fraud_rate
FROM #base
WHERE trans_dt < '2020-01-01'
GROUP BY merchant;

CREATE UNIQUE CLUSTERED INDEX CIX_mfr ON #merch_fraud_rate (merchant);
GO




-- STEP 8: Build dbo.features_sql
IF OBJECT_ID('dbo.features_sql', 'U') IS NOT NULL
    DROP TABLE dbo.features_sql;

SELECT

    -- Keys + raw fields
    b.trans_num,
    b.cc_num,
    b.merchant,
    b.is_fraud,
    b.unix_time,
    b.cust_lat,
    b.cust_long,
    b.merch_lat,
    b.merch_long,

    -- CALENDAR
    b.TransactionHour,
    b.DayOfWeek,
    b.IsWeekend,
    b.IsNightTxn,

    -- AMOUNT
    b.amt,
    CAST(
        CASE WHEN cs.cust_std_amt > 0
             THEN (b.amt - cs.cust_avg_amt) / cs.cust_std_amt
             ELSE 0.0 END
    AS FLOAT) AS AmountZScore,
    ar.AmountPercentileRank,
    CASE WHEN b.amt > cs.cust_avg_amt * 2.0 THEN 1 ELSE 0 END AS IsHighAmount,
    CAST(b.amt / NULLIF(cs.cust_max_amt, 0) AS FLOAT) AS AmtToMaxRatio,

    -- VELOCITY
    v.TxnCount_1h,
    v.TxnCount_6h,
    v.TxnCount_24h,
    v.TxnCount_7d,
    CAST(v.TxnAmount_1h AS FLOAT) AS TxnAmount_1h,
    CAST(v.TxnAmount_24h AS FLOAT) AS TxnAmount_24h,
    CAST(v.TxnAmount_7d AS FLOAT) AS TxnAmount_7d,
    CAST(
        CASE WHEN v.TxnCount_24h > 0
             THEN v.TxnAmount_24h / v.TxnCount_24h
             ELSE 0.0 END
    AS FLOAT) AS AvgTxnAmtLast24h,

    -- SEQUENTIAL
    CAST(seq.TimeSinceLastTxnSec AS BIGINT) AS TimeSinceLastTxnSec,
    CASE WHEN seq.TimeSinceLastTxnSec IS NULL THEN 1 ELSE 0 END AS IsFirstTxn,
    CASE WHEN seq.MerchantVisitRank = 1 THEN 1 ELSE 0 END AS IsFirstTimeAtMerchant,
    seq.MerchantVisitRank,
    CAST(COALESCE(seq.LatDeltaFromLastTxn, 0) AS FLOAT) AS LatDeltaFromLastTxn,
    CAST(COALESCE(seq.LongDeltaFromLastTxn, 0) AS FLOAT) AS LongDeltaFromLastTxn,
    COALESCE(seq.SuspiciousGeoSpeed, 0) AS SuspiciousGeoSpeed,

    -- LIFECYCLE
    DATEDIFF(DAY, ce.AccountCreatedAt, b.trans_dt) AS DaysSinceAccountCreation,
    DATEDIFF(DAY, cs.cust_first_txn_date, CAST(b.trans_dt AS DATE)) AS DaysSinceFirstTxn,
    DATEDIFF(DAY, pc.LastChangeDate, b.trans_dt) AS DaysSincePasswordChange,

    -- DIVERSITY
    cs.TotalDistinctMerchants,
    cs.TotalDistinctCategories,

    -- IDENTITY / CUSTOMER
    COALESCE(ce.AccountAgeDays, 0) AS AccountAgeDays,
    COALESCE(ce.IdentityConfidenceScore, 50.0) AS IdentityConfidenceScore,
    COALESCE(ce.AddressConsistencyScore, 50.0) AS AddressConsistencyScore,
    COALESCE(ce.PreviousFraudFlag, 0) AS PreviousFraudFlag,
    CASE WHEN ce.EmailDomainType = 'disposable' THEN 1 ELSE 0 END AS IsDisposableEmailDomain,

    -- EMAIL INTELLIGENCE
    COALESCE(ei.EmailRiskScore, 0.0) AS EmailRiskScore,
    COALESCE(ei.IsDisposableEmail, 0) AS IsDisposableEmail,
    COALESCE(ei.EmailAgeDays, 0) AS EmailAgeDays,
    COALESCE(ei.DomainReputationScore, 1.0) AS DomainReputationScore,

    -- PHONE INTELLIGENCE
    COALESCE(phi.PhoneRiskScore, 0.0) AS PhoneRiskScore,
    COALESCE(phi.IsVOIP, 0) AS IsVOIP,
    COALESCE(phi.PhoneAgeDays, 0) AS PhoneAgeDays,
    COALESCE(phi.CarrierType, 'unknown') AS CarrierType,

    -- IP / NETWORK
    COALESCE(ip.IPRiskScore, 0.0) AS IPRiskScore,
    COALESCE(ip.IsVPN, 0) AS IsVPN,
    COALESCE(ip.IsProxy, 0) AS IsProxy,
    COALESCE(ip.IsTor, 0) AS IsTor,
    COALESCE(ip.IPCountryMismatch, 0) AS IPCountryMismatch,
    COALESCE(ip.IP_Country, 'US') AS IP_Country,

    -- ADDRESS INTELLIGENCE
    COALESCE(ai.AddressRiskScore, 0.0) AS AddressRiskScore,
    COALESCE(ai.IsCommercialAddress, 0) AS IsCommercialAddress,
    COALESCE(ai.GeoConsistencyScore, 1.0) AS GeoConsistencyScore,

    -- IDENTITY BUREAU
    COALESCE(ib.SyntheticIdentityFlag, 0) AS SyntheticIdentityFlag,
    COALESCE(ib.SSNMatchScore, 1.0) AS SSNMatchScore,
    COALESCE(ib.IdentityVerificationStatus, 'Unknown') AS IdentityVerificationStatus,
    CASE ib.CreditScoreRange
        WHEN 'poor' THEN 0
        WHEN 'fair' THEN 1
        WHEN 'good' THEN 2
        WHEN 'very good' THEN 3
        WHEN 'exceptional' THEN 4
        ELSE 0
    END AS CreditScoreOrdinal,

    -- DEVICE / SESSION
    COALESCE(sl.IsNewDevice, 0) AS IsNewDevice,
    COALESCE(sl.BotLikelihoodScore, 0.0) AS BotLikelihoodScore,
    COALESCE(sl.CheckoutTimeSec, 0) AS CheckoutTimeSec,
    COALESCE(sl.SessionDurationSec, 0) AS SessionDurationSec,
    COALESCE(sl.PagesViewed, 0) AS PagesViewed,
    COALESCE(sl.InteractionComplexityScore, 0.0) AS InteractionComplexityScore,
    COALESCE(dt.TrustScore, 50.0) AS DeviceTrustScore,
    COALESCE(dt.NumLinkedAccounts, 1) AS NumAccountsOnDevice,
    COALESCE(dt.ChangeFrequency, 0.0) AS DeviceChangeFrequency,
    COALESCE(dt.BlacklistFlag, 0) AS DeviceBlacklisted,

    -- MERCHANT
    COALESCE(me.MerchantRiskScore, 0.0) AS MerchantRiskScore,
    COALESCE(me.MerchantFraudRate, 0.0) AS MerchantFraudRate,
    -- Train-period merchant fraud rate (no test leakage)
    COALESCE(mfr.train_fraud_rate, me.MerchantFraudRate, 0.0) AS MerchantFraudRate_Train,
    COALESCE(me.IsHighRiskCategory, 0) AS IsHighRiskCategory,
    COALESCE(me.MerchantCategory, b.category) AS MerchantCategory,
    COALESCE(me.MerchantCountry, 'US') AS MerchantCountry,

    -- ACCOUNT HISTORY
    COALESCE(ah.ChargebackCount, 0) AS ChargebackCount,
    COALESCE(ah.FraudFlagHistory, 0) AS FraudFlagHistory,
    COALESCE(ah.VelocityScore, 0.0) AS HistoricalVelocityScore,
    CAST(
        CASE WHEN COALESCE(ah.TotalTransactions, 0) > 0
             THEN CAST(ah.ChargebackCount AS FLOAT) / ah.TotalTransactions
             ELSE 0.0 END
    AS FLOAT) AS ChargebackRatio,

    -- LOGIN BEHAVIOR
    COALESCE(la.LoginFailureRate, 0.0) AS LoginFailureRate,
    COALESCE(la.TotalLoginAttempts, 0) AS TotalLoginAttempts,

    -- RULE-BASED FLAGS
    CASE WHEN v.TxnCount_1h >= 5 THEN 1 ELSE 0 END AS Rule_HighVelocity,
    CASE WHEN COALESCE(ip.IPCountryMismatch, 0) = 1 THEN 1 ELSE 0 END AS Rule_CountryMismatch,
    CASE WHEN COALESCE(sl.IsNewDevice, 0) = 1 AND b.amt > cs.cust_avg_amt * 2.0 THEN 1 ELSE 0 END AS Rule_NewDeviceHighAmount,
    CASE WHEN COALESCE(ei.IsDisposableEmail, 0) = 1 THEN 1 ELSE 0 END AS Rule_DisposableEmail,
    CASE WHEN COALESCE(ip.IsVPN, 0) = 1 OR COALESCE(ip.IsTor, 0) = 1 THEN 1 ELSE 0 END AS Rule_VPNorTor,
    CASE WHEN COALESCE(ib.SyntheticIdentityFlag, 0) = 1 THEN 1 ELSE 0 END AS Rule_SyntheticIdentity,
    CASE WHEN COALESCE(ah.ChargebackCount, 0) >= 3 THEN 1 ELSE 0 END AS Rule_HighChargebacks,
    CASE WHEN COALESCE(sl.BotLikelihoodScore, 0.0) >= 0.7 THEN 1 ELSE 0 END AS Rule_BotLikely,
    CASE WHEN DATEDIFF(DAY, ce.AccountCreatedAt, b.trans_dt) <= 7 THEN 1 ELSE 0 END AS Rule_NewAccount,
    CASE WHEN COALESCE(phi.IsVOIP, 0) = 1 THEN 1 ELSE 0 END AS Rule_VOIPPhone,

    -- COMPOSITE RISK COUNT
    (
        CASE WHEN v.TxnCount_1h >= 5 THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(ip.IPCountryMismatch, 0) = 1 THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(sl.IsNewDevice, 0) = 1 AND b.amt > cs.cust_avg_amt * 2 THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(ei.IsDisposableEmail, 0) = 1 THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(ip.IsVPN, 0) = 1 OR COALESCE(ip.IsTor, 0) = 1 THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(ib.SyntheticIdentityFlag, 0) = 1 THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(ah.ChargebackCount, 0) >= 3 THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(sl.BotLikelihoodScore, 0.0) >= 0.7 THEN 1 ELSE 0 END
      + CASE WHEN DATEDIFF(DAY, ce.AccountCreatedAt, b.trans_dt) <= 7 THEN 1 ELSE 0 END
      + CASE WHEN COALESCE(phi.IsVOIP, 0) = 1 THEN 1 ELSE 0 END
    ) AS RiskSignalCount

INTO dbo.features_sql

FROM #base b
JOIN #cust_stats cs ON cs.cc_num = b.cc_num
JOIN #velocity v ON v.trans_num = b.trans_num
JOIN #sequential seq ON seq.trans_num = b.trans_num
JOIN #amt_rank ar ON ar.trans_num = b.trans_num

LEFT JOIN #pwd_change pc ON pc.cc_num = b.cc_num
LEFT JOIN #login_agg la ON la.cc_num = b.cc_num
LEFT JOIN dbo.customers_extended ce ON ce.cc_num = b.cc_num
LEFT JOIN dbo.email_intel_feed ei ON ei.cc_num = b.cc_num
LEFT JOIN dbo.phone_intel_feed phi ON phi.cc_num = b.cc_num
LEFT JOIN dbo.ip_intel_feed ip ON ip.trans_num = b.trans_num
LEFT JOIN dbo.address_intel_feed ai ON ai.cc_num = b.cc_num
LEFT JOIN dbo.identity_bureau_feed ib ON ib.cc_num = b.cc_num
LEFT JOIN dbo.session_logs sl ON sl.trans_num = b.trans_num
LEFT JOIN dbo.device_trust_feed dt ON dt.DeviceID = sl.DeviceID
LEFT JOIN dbo.merchants_extended me ON me.merchant = b.merchant
LEFT JOIN #merch_fraud_rate mfr ON mfr.merchant = b.merchant
LEFT JOIN dbo.account_history ah ON ah.cc_num = b.cc_num;
GO




-- STEP 9: Indexes on features_sql for fast downstream queries
CREATE CLUSTERED INDEX CIX_features_trans_num
    ON dbo.features_sql (trans_num);
GO

CREATE NONCLUSTERED INDEX IX_features_cc_num
    ON dbo.features_sql (cc_num)
    INCLUDE (is_fraud, AmountZScore, TxnCount_24h, RiskSignalCount);
GO

CREATE NONCLUSTERED INDEX IX_features_fraud_label
    ON dbo.features_sql (is_fraud)
    INCLUDE (trans_num, RiskSignalCount)
    WHERE is_fraud IS NOT NULL;
GO




-- STEP 10: Validation summary
SELECT
    COUNT(*) AS TotalRows,
    SUM(is_fraud) AS FraudRows,
    CAST(SUM(is_fraud) AS FLOAT) / COUNT(*) AS FraudRate,
    COUNT(DISTINCT cc_num) AS UniqueCustomers,
    AVG(CAST(RiskSignalCount AS FLOAT)) AS AvgRiskSignals,
    -- Fraud vs legit signal separation
    AVG(CASE WHEN is_fraud = 1 THEN TxnCount_1h ELSE NULL END) AS Fraud_AvgTxnCount_1h,
    AVG(CASE WHEN is_fraud = 0 THEN TxnCount_1h ELSE NULL END) AS Legit_AvgTxnCount_1h,
    AVG(CASE WHEN is_fraud = 1 THEN AmountZScore ELSE NULL END) AS Fraud_AvgAmountZScore,
    AVG(CASE WHEN is_fraud = 0 THEN AmountZScore ELSE NULL END) AS Legit_AvgAmountZScore,
    AVG(CASE WHEN is_fraud = 1 THEN IPRiskScore ELSE NULL END) AS Fraud_AvgIPRiskScore,
    AVG(CASE WHEN is_fraud = 0 THEN IPRiskScore ELSE NULL END) AS Legit_AvgIPRiskScore,
    AVG(CASE WHEN is_fraud = 1 THEN BotLikelihoodScore ELSE NULL END) AS Fraud_AvgBotScore,
    AVG(CASE WHEN is_fraud = 0 THEN BotLikelihoodScore ELSE NULL END) AS Legit_AvgBotScore,
    -- Rule fire rates
    CAST(SUM(Rule_HighVelocity) AS FLOAT) / COUNT(*) AS FireRate_HighVelocity,
    CAST(SUM(Rule_VPNorTor) AS FLOAT) / COUNT(*) AS FireRate_VPNorTor,
    CAST(SUM(Rule_BotLikely) AS FLOAT) / COUNT(*) AS FireRate_BotLikely,
    CAST(SUM(Rule_NewDeviceHighAmount) AS FLOAT) / COUNT(*) AS FireRate_NewDeviceHighAmt
FROM dbo.features_sql;
GO
