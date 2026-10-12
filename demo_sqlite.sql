-- Version 3 demo schema: stable IDs and price observation times.
-- Do not rewrite existing databases.
-- Fictional data; offer observation times are set when the DB is created.
-- These rows do not establish VIN compatibility.
PRAGMA foreign_keys = ON;

CREATE TABLE vehicles (
    vehicle_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    manufacturer TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER NOT NULL,
    engine TEXT NOT NULL
);

CREATE TABLE parts (
    part_number TEXT PRIMARY KEY,
    vehicle_id TEXT NOT NULL REFERENCES vehicles(vehicle_id),
    category TEXT NOT NULL,
    brand_id TEXT NOT NULL,
    position TEXT NOT NULL
);

CREATE TABLE offers (
    part_number TEXT NOT NULL REFERENCES parts(part_number),
    seller_id TEXT NOT NULL,
    price_rub INTEGER NOT NULL CHECK (price_rub >= 0),
    in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    observed_at TEXT NOT NULL,
    PRIMARY KEY (part_number, seller_id)
);

INSERT INTO vehicles VALUES
    ('1', '42', 'Opel', 'Corsa D', 2007, 'Z12XEP'),
    ('2', '43', 'Demo Motors', 'City', 2010, 'D-100');

INSERT INTO parts VALUES
    ('13.0460-7117.2', '1', 'front brake pads', 'brand_ate', 'front axle'),
    ('P 59 045', '1', 'front brake pads', 'brand_brembo', 'front axle'),
    ('DEMO-FP-2', '2', 'front brake pads', 'brand_demoline', 'front axle'),
    ('DEMO-RD-2', '2', 'rear brake discs', 'brand_demoline', 'rear axle');

INSERT INTO offers VALUES
    ('13.0460-7117.2', 'seller_demo_parts_store', 4900, 1, strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    ('P 59 045', 'seller_demo_parts_store', 4350, 1, strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    ('DEMO-RD-2', 'seller_demo_parts_store', 6200, 0, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
