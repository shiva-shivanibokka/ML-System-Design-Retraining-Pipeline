# DVC + DagsHub Remote Setup

DVC has been initialized in this repository (`dvc init`), which created `.dvc/config`,
`.dvc/.gitignore`, and `.dvcignore`. No remote is configured yet.

## Configuring the DagsHub remote

Once a DagsHub repository exists for this project, run the following commands **once**
to point DVC at it (replace `<user>/<repo>` and `<DAGSHUB_TOKEN>` with real values):

```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify origin --local access_key_id <DAGSHUB_TOKEN>
dvc remote modify origin --local secret_access_key <DAGSHUB_TOKEN>
```

Notes:

- The `--local` flag writes credentials to `.dvc/config.local`, which is gitignored
  (via `.dvc/.gitignore`) and never committed.
- A DagsHub personal access token can be used for both `access_key_id` and
  `secret_access_key`.
- These commands are documented here for reference only — they have not been executed,
  since no DagsHub repository/credentials exist yet.

## Next steps

Task 0.6 performs the first `dvc add` (to version the cleaned dataset) and `dvc push`
(to upload it to the DagsHub remote configured above).
