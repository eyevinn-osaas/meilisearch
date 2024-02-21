use std::fs::File as StdFile;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use tempfile::NamedTempFile;
use uuid::Uuid;

const UPDATE_FILES_PATH: &str = "updates/updates_files";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Could not parse file name as utf-8")]
    CouldNotParseFileNameAsUtf8,
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    PersistError(#[from] tempfile::PersistError),
    #[error(transparent)]
    UuidError(#[from] uuid::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Deref for File {
    type Target = NamedTempFile;

    fn deref(&self) -> &Self::Target {
        &self.file
    }
}

impl DerefMut for File {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.file
    }
}

#[derive(Clone, Debug)]
pub struct FileStore {
    path: PathBuf,
}

impl FileStore {
    pub fn new(path: impl AsRef<Path>) -> Result<FileStore> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;
        Ok(FileStore { path })
    }
}

impl FileStore {
    /// Creates a new temporary update file.
    /// A call to `persist` is needed to persist the file in the database.
    pub fn new_update(&self) -> Result<(Uuid, File)> {
        let file = NamedTempFile::new_in(&self.path)?;
        let uuid = Uuid::new_v4();
        let path = self.path.join(uuid.to_string());
        let update_file = File { dry: false, file, path };

        Ok((uuid, update_file))
    }

    /// Creates a new temporary update file with the given Uuid.
    /// A call to `persist` is needed to persist the file in the database.
    pub fn new_update_with_uuid(&self, uuid: u128) -> Result<(Uuid, File)> {
        let file = NamedTempFile::new_in(&self.path)?;
        let uuid = Uuid::from_u128(uuid);
        let path = self.path.join(uuid.to_string());
        let update_file = File { dry: false, file, path };

        Ok((uuid, update_file))
    }

    /// Returns the file corresponding to the requested uuid.
    pub fn get_update(&self, uuid: Uuid) -> Result<StdFile> {
        let path = self.get_update_path(uuid);
        let file = StdFile::open(path)?;
        Ok(file)
    }

    /// Returns the path that correspond to this uuid, the path could not exists.
    pub fn get_update_path(&self, uuid: Uuid) -> PathBuf {
        self.path.join(uuid.to_string())
    }

    /// Copies the content of the update file pointed to by `uuid` to the `dst` directory.
    pub fn snapshot(&self, uuid: Uuid, dst: impl AsRef<Path>) -> Result<()> {
        let src = self.path.join(uuid.to_string());
        let mut dst = dst.as_ref().join(UPDATE_FILES_PATH);
        std::fs::create_dir_all(&dst)?;
        dst.push(uuid.to_string());
        std::fs::copy(src, dst)?;
        Ok(())
    }

    /// Compute the size of all the updates contained in the file store.
    pub fn compute_total_size(&self) -> Result<u64> {
        let mut total = 0;
        for uuid in self.all_uuids()? {
            total += self.compute_size(uuid?).unwrap_or_default();
        }
        Ok(total)
    }

    /// Compute the size of one update
    pub fn compute_size(&self, uuid: Uuid) -> Result<u64> {
        Ok(self.get_update(uuid)?.metadata()?.len())
    }

    pub fn delete(&self, uuid: Uuid) -> Result<()> {
        let path = self.path.join(uuid.to_string());
        std::fs::remove_file(path)?;
        Ok(())
    }

    /// List the Uuids of the files in the FileStore
    pub fn all_uuids(&self) -> Result<impl Iterator<Item = Result<Uuid>>> {
        Ok(self.path.read_dir()?.filter_map(|entry| {
            let file_name = match entry {
                Ok(entry) => entry.file_name(),
                Err(e) => return Some(Err(e.into())),
            };
            let file_name = match file_name.to_str() {
                Some(file_name) => file_name,
                None => return Some(Err(Error::CouldNotParseFileNameAsUtf8)),
            };
            if file_name.starts_with('.') {
                None
            } else {
                Some(Uuid::from_str(file_name).map_err(|e| e.into()))
            }
        }))
    }
}

pub struct File {
    dry: bool,
    path: PathBuf,
    file: NamedTempFile,
}

impl File {
    pub fn dry_file() -> Result<Self> {
        #[cfg(target_family = "unix")]
        let path = PathBuf::from_str("/dev/null").unwrap();
        #[cfg(target_family = "windows")]
        let path = PathBuf::from_str("\\Device\\Null").unwrap();

        Ok(Self {
            dry: true,
            path: path.clone(),
            file: tempfile::Builder::new().make(|_| std::fs::File::create(path.clone()))?,
        })
    }

    pub fn persist(self) -> Result<()> {
        if !self.dry {
            self.file.persist(&self.path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::io::Write;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn all_uuids() {
        let dir = TempDir::new().unwrap();
        let fs = FileStore::new(dir.path()).unwrap();
        let (uuid, mut file) = fs.new_update().unwrap();
        file.write_all(b"Hello world").unwrap();
        file.persist().unwrap();
        let all_uuids = fs.all_uuids().unwrap().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(all_uuids, vec![uuid]);

        let (uuid2, file) = fs.new_update().unwrap();
        let all_uuids = fs.all_uuids().unwrap().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(all_uuids, vec![uuid]);

        file.persist().unwrap();
        let mut all_uuids = fs.all_uuids().unwrap().collect::<Result<Vec<_>>>().unwrap();
        all_uuids.sort();
        let mut expected = vec![uuid, uuid2];
        expected.sort();
        assert_eq!(all_uuids, expected);
    }
}
