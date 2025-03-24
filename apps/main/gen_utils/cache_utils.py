import json
import hashlib
import sqlite3
import time
from tqdm import tqdm
from collections import defaultdict
from functools import wraps
import copy

def retry_on_locked(max_retries=5, initial_delay=0.5, backoff_factor=2, fallback_return=None):
    """
    Decorator to retry a function upon encountering a 'database is locked' error.
    
    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Initial delay between retries in seconds.
        backoff_factor (int or float): Multiplier for delay after each retry.
        fallback_return (any): The value to return if the function fails after max_retries. If None, the function will raise the last encountered error.
    Returns:
        Decorated function with retry logic.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        if fallback_return is not None:
                            return fallback_return
                        else:
                            raise e
        return wrapper
    return decorator


class SQLiteLockConnection:
    def __init__(self, db_path, keys_type_dict, values_type_dict, timeout=300):
        """
        Initializes the SQLiteLockConnection.

        Args:
            db_path (str): Path to the SQLite database file.
            keys_type_dict (dict): Dictionary mapping key names to their SQLite types.
            values_type_dict (dict): Dictionary mapping value names to their SQLite types.
            timeout (int, optional): SQLite connection timeout in seconds. Defaults to 30.
        """
        self.db_path = db_path
        self.timeout = timeout  # SQLite connection timeout in seconds
        self.keys_type_dict = keys_type_dict
        self.values_type_dict = values_type_dict
        self.conn = self._get_connection()
        self._optimize_connection()
        # self._enable_wal_mode()
        self._create_db()

    def _get_connection(self):
        """
        Establishes a persistent SQLite connection with the specified timeout.

        Returns:
            sqlite3.Connection: SQLite connection object.
        """
        return sqlite3.connect(self.db_path, timeout=self.timeout, check_same_thread=False)

    def _optimize_connection(self):
        """
        Optimizes SQLite connection for better read performance.
        """
        with self.conn:
            # Set journal mode to memory for faster transactions
            self.conn.execute('PRAGMA journal_mode = MEMORY')
            # Disable synchronous writes completely since we're operating in memory
            self.conn.execute('PRAGMA synchronous = OFF')
            self.conn.execute('PRAGMA cache_size = -8000000')  # 8GB
            # Enable memory-mapped I/O for better read performance
            self.conn.execute('PRAGMA mmap_size = 8000000000')  # 8GB
            self.conn.execute('PRAGMA temp_store = MEMORY')  # Store temporary tables in memory

    # @retry_on_locked(max_retries=10, initial_delay=0.5, backoff_factor=2)
    # def _enable_wal_mode(self):
    #     """
    #     Enables Write-Ahead Logging (WAL) mode to improve concurrency.
    #     """
    #     with self.conn:
    #         self.conn.execute('PRAGMA journal_mode = WAL')
    #         # PRAGMA busy_timeout doesn't accept parameters, needs direct value
    #         self.conn.execute(f'PRAGMA busy_timeout = {self.timeout * 1000}')

    @retry_on_locked(max_retries=10, initial_delay=1, backoff_factor=2)
    def _create_db(self):
        """
        Creates the 'items' table and an index on 'key_hash' if they do not exist.
        """
        with self.conn:
            c = self.conn.cursor()
            columns = (
                [f"{key} {self.keys_type_dict[key]}" for key in self.keys_type_dict] +
                [f"{key} {self.values_type_dict[key]}" for key in self.values_type_dict]
            )
            create_table_sql = f'''CREATE TABLE IF NOT EXISTS items
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          key_hash TEXT,
                          {', '.join(columns)})'''
            c.execute(create_table_sql)
            c.execute('CREATE INDEX IF NOT EXISTS idx_key_hash ON items(key_hash)')

    def _get_key_hash(self, key_dict):
        """
        Generates an MD5 hash for the given key dictionary.

        Args:
            key_dict (dict): Dictionary of keys.

        Returns:
            str: MD5 hash of the serialized key dictionary.
        """
        key_json = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()
    
    @retry_on_locked(max_retries=10, initial_delay=0.5, backoff_factor=2)
    def get_items_batch(self, key_dicts, limit=None, value_only=False, delete=False, return_ids=False, batch_size=1000):
        """
        Retrieves multiple items in batch from the database based on provided key dictionaries.

        Args:
            key_dicts (list): List of key dictionaries to filter items.
            limit (int, optional): Maximum number of items to retrieve per key_dict.
            value_only (bool, optional): If True, retrieves only value columns.
            delete (bool, optional): If True, deletes the retrieved items.
            return_ids (bool, optional): If True, includes item IDs in results.
            batch_size (int, optional): Number of key_hashes to process in each batch.

        Returns:
            list: List of items corresponding to the order of key_dicts.
        """
        results = [[]] * len(key_dicts) 
        key_hash_to_indices = defaultdict(list)  # Maps key_hash to list of indices in key_dicts
        
        # Pre-compute all hashes and handle duplicates
        key_hashes = []
        for index, key_dict in enumerate(key_dicts):
            key_hash = self._get_key_hash(key_dict)
            if key_hash not in key_hash_to_indices:
                key_hashes.append(key_hash)
            key_hash_to_indices[key_hash].append(index)
        
        # Process in batches
        for i in tqdm(range(0, len(key_hashes), batch_size), desc="Processing cache query batches"):
            batch_hashes = key_hashes[i:i + batch_size]
            placeholders = ','.join(['?' for _ in batch_hashes])
            
            c = self.conn.cursor()
            if limit is None:
                query = f"SELECT * FROM items WHERE key_hash IN ({placeholders})"
                c.execute(query, batch_hashes)
            else:
                query = f"""
                    SELECT * FROM (
                        SELECT *, 
                            ROW_NUMBER() OVER (PARTITION BY key_hash) as rn 
                        FROM items 
                        WHERE key_hash IN ({placeholders})
                    ) sub 
                    WHERE rn <= ?
                """
                c.execute(query, batch_hashes + [limit])
            
            batch_results = c.fetchall()
            column_names = [description[0] for description in c.description]
            
            # Group results by key_hash
            grouped_results = {}
            for row in batch_results:
                key_hash = row[column_names.index('key_hash')]
                if key_hash not in grouped_results:
                    grouped_results[key_hash] = []
                grouped_results[key_hash].append(row)
            
            # Delete if requested
            if delete and batch_results:
                ids_to_delete = [row[0] for row in batch_results]
                delete_placeholders = ','.join(['?' for _ in ids_to_delete])
                c.execute(f"DELETE FROM items WHERE id IN ({delete_placeholders})", ids_to_delete)
                self.conn.commit()
            
            # Format results
            exclude_columns = []
            if not return_ids:
                exclude_columns.append('id')
            if value_only:
                exclude_columns.append('key_hash')
                exclude_columns.extend(list(self.keys_type_dict.keys()))
            if 'rn' in column_names:
                exclude_columns.append('rn')
            
            # Add formatted results to main results list, handling duplicates
            for key_hash, rows in grouped_results.items():
                formatted_rows = [
                    {column: value for column, value in zip(column_names, row) 
                     if column not in exclude_columns}
                    for row in rows
                ]
                # Assign same results to all indices that generated this hash
                for j, index in enumerate(key_hash_to_indices[key_hash]):
                    results[index] = copy.deepcopy(formatted_rows) if j > 0 else formatted_rows  # deepcopy only if there are duplicates
            
        return results

    def get_items(self, key_dict, limit=None, value_only=False, delete=False, return_ids=False):
        """
        Backward-compatible wrapper for single key_dict queries.
        """
        results = self.get_items_batch(
            [key_dict], 
            limit=limit, 
            value_only=value_only, 
            delete=delete, 
            return_ids=return_ids,
            batch_size=1,
        )
        return results[str(key_dict)]

    @retry_on_locked(max_retries=10, initial_delay=0.5, backoff_factor=2)
    def add_items(self, items):
        """
        Adds multiple items to the database.

        Args:
            items (list): List of dictionaries with 'key_dict' and 'value_dict' keys.
        """
        key_columns = list(self.keys_type_dict.keys())
        value_columns = list(self.values_type_dict.keys())
        columns = key_columns + value_columns
        placeholders = ', '.join(['?' for _ in range(len(columns) + 1)])  # +1 for key_hash 
        
        c = self.conn.cursor()
        for item in items:
            key_dict, value_dict = item["key_dict"], item["value_dict"]
            key_hash = self._get_key_hash(key_dict)
            c.execute(f'''
                INSERT INTO items (key_hash, {', '.join(columns)})
                VALUES ({placeholders})
            ''', (key_hash, *[key_dict[k] for k in key_columns], *[value_dict[k] for k in value_columns]))
        self.conn.commit()

    @retry_on_locked(max_retries=10, initial_delay=0.5, backoff_factor=2)
    def update_items(self, items, ids):
        """
        Updates multiple items in the database.

        Args:
            items (list): List of dictionaries with 'key_dict' and 'value_dict' keys.
            ids (list): List of item IDs corresponding to the items to update.

        Raises:
            ValueError: If the number of items does not match the number of IDs.
        """
        if len(items) != len(ids):
            raise ValueError("Number of items must match number of IDs")
            
        key_columns = list(self.keys_type_dict.keys())
        value_columns = list(self.values_type_dict.keys())
        columns = key_columns + value_columns
        
        # Create SET clause for UPDATE statement
        set_clause = ', '.join([f"{col} = ?" for col in columns])
        
        c = self.conn.cursor()
        for item, item_id in zip(items, ids):
            key_dict, value_dict = item["key_dict"], item["value_dict"]
            key_hash = self._get_key_hash(key_dict)
            
            # Update both key_hash and all columns
            c.execute(f'''
                UPDATE items 
                SET key_hash = ?, {set_clause}
                WHERE id = ?
            ''', (key_hash, *[key_dict[k] for k in key_columns], 
                  *[value_dict[k] for k in value_columns], item_id))
        self.conn.commit()

    def close(self):
        """
        Closes the SQLite connection.
        """
        self.conn.close()

if __name__ == "__main__":
    cache = SQLiteLockConnection(
        db_path="test.db",
        keys_type_dict={"firstname": "TEXT", "lastname": "TEXT"},
        values_type_dict={"age": "INTEGER", "height": "REAL"}
    )
    cache.add_items([
        {
            "key_dict": {"firstname": "John", "lastname": "Doe"},
            "value_dict": {"age": 20, "height": 170.5}
        }
    ])
    print(cache.get_items({"firstname": "John", "lastname": "Doe"}))
    cache.close()