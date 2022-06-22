import paramiko
import os
from pathlib import Path
from stat import S_ISDIR
from tqdm.autonotebook import tqdm
from core_io.print_msg import *


class SFTPClient:
    """ SFTP Client for remote file management.
    """
    
    def __init__(self, host_addr, user_name, pub_key_path='id_rsa', port=22, passwd=None, auto_connect=True):
        self.host_addr = host_addr
        self.port = port
        self.user_name = user_name
        self.password = passwd
        self.pub_key_path = pub_key_path

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        if auto_connect:
            self.connect()
    
    def connect(self):
        if self.pub_key_path is not None:
            pub_key_path = Path.home() / '.ssh' / self.pub_key_path if self.pub_key_path == 'id_rsa'else Path(self.pub_key_path)

            if not pub_key_path.exists():
                raise Exception("key file %s not exists" % pub_key_path)
            k = paramiko.RSAKey.from_private_key_file(pub_key_path)
            self.ssh.connect(self.host_addr, self.port, username=self.user_name, pkey=k)
        elif self.password is not None:
            self.ssh.connect(self.host_addr, self.port, username=self.user_name, password=self.password)
        else:
            raise Exception("SSH Connection can not be established")

        self.sftp = self.ssh.open_sftp()

    def disconnect(self):
        if self.ssh:
            self.ssh.close()
        if self.sftp:
            self.sftp.close()
    
    def __del__(self, ):
        self.disconnect()

    def list_remote_dir(self, remote_dir, show_hiddens=False):
        """ List remote dir
        """
        items = self.sftp.listdir(remote_dir)
        items = [item for item in items] if show_hiddens is True else [item for item in items if not item.startswith('.')]
        return items

    def remote_exists(self, remote_file_path: str) -> bool:
        """ Check if remote file or directory is exists.
        """
        try:
            self.sftp.stat(remote_file_path)
            return True
        except IOError:
            return False

    def is_dir(self, remote_path):
        """ Check if the remote path is a directory.
        """
        try:
            return S_ISDIR(self.sftp.stat(remote_path).st_mode)
        except IOError:
            # Path does not exist, so by definition not a directory
            return False

    def recursive_list_local_files(self, local_dir, exclude_folder=True, ignore_links=True):
        """
            Recursively visiting local files

        Args:
            local_dir (str or Path): local dir to be visited.
            exclude_folder (bool): exclude folders in visited file list.
            ignore_links (bool): ignore the links when iterating.

        Returns:
            directory list and file list

        """
        queue = os.listdir(str(local_dir))

        item_list = []
        dir_list = []
        while len(queue) > 0:
            item = queue.pop(0)
            full_path = os.path.join(local_dir, item)

            if os.path.islink(full_path) and ignore_links is True:
                continue

            if os.path.isdir(full_path):
                try:
                    dl = os.listdir(full_path)
                except PermissionError:
                    continue
                queue += [os.path.join(item, sub_item) for sub_item in dl]
                dir_list.append(item)

                if exclude_folder is False:
                    item_list.append(item)
            elif exclude_folder:
                item_list.append(item)

        return dir_list, item_list

    def to_remote(self, from_local, to_remote, show_progress=False):
        """
            Copy files or directory to remote

        Args:
            from_local (str or Path): local file path or directory
            to_remote (str or Path): remote file path or directory
            show_progress (bool): display progress bar

        """
        from_local = Path(from_local)
        if not from_local.exists():
            raise Exception('Local path or dir %s not exists' % from_local)

        if from_local.is_dir():
            local_dir_list, local_file_list = self.recursive_list_local_files(from_local)

            # create remote directories
            for dir_ in local_dir_list:
                remote_dir = os.path.join(to_remote, dir_)
                if not self.remote_exists(remote_dir):
                    SFTPClient.__mkdir_p__(self.sftp, remote_dir)

            for item in tqdm(local_file_list, desc=msg('uploading', self, True), disable=not show_progress):
                self.sftp.put(os.path.join(from_local, item), os.path.join(str(to_remote), item), )
            
        else:
            self.__put_file__(str(from_local), str(to_remote))

    def recursive_list_remote_files(self, remote_dir, exclude_folder=True):
        """
            Recursively visiting remote files

        Args:
            remote_dir (str or Path): remote dir to be visited.
            exclude_folder (bool): exclude folders in visited file list.

        Returns:
            directory list and file list

        """
        queue = self.sftp.listdir(remote_dir)

        item_list = []
        dir_list = []
        while len(queue) > 0:
            item = queue.pop(0)
            full_path = os.path.join(remote_dir, item)

            if self.is_dir(full_path):
                dl = self.sftp.listdir(full_path)
                queue += [os.path.join(item, sub_item) for sub_item in dl]
                dir_list.append(item)

                if exclude_folder is False:
                    item_list.append(item)
            elif exclude_folder:
                item_list.append(item)

        return dir_list, item_list

    def from_remote(self, from_remote, to_local, show_progress=False):
        """
            Copy files or directory from remote to local

        Args:
            from_remote (str or Path): remote file path or directory
            to_local (str or Path): local file path or directory
            show_progress (bool): display progress bar

        """
        if not self.remote_exists(from_remote):
            raise Exception('Remote path %s not exists' % from_remote)
            
        to_local = Path(to_local)

        if self.is_dir(from_remote):

            if not to_local.is_dir():
                if not to_local.exists():
                    to_local.mkdir(parents=True, exist_ok=False)
                else:
                    raise Exception('Local dir %s should be a directory' % to_local)

            # gathering remote files
            remote_dir_list, remote_file_list = self.recursive_list_remote_files(from_remote, exclude_folder=True)

            # create local directories
            for dir_ in remote_dir_list:
                local_dir = (to_local / dir_)
                if not local_dir.exists():
                    local_dir.mkdir(parents=True, exist_ok=False)

            for item in tqdm(remote_file_list, desc=msg('downloading', self, True), disable=not show_progress):
                self.sftp.get(os.path.join(str(from_remote), item), os.path.join(to_local, item))
            
            return True
        elif self.remote_exists(from_remote):
            self.__get_file__(from_remote, to_local)
            return True
        else:
            return False

    def __put_file__(self, from_local_, to_remote):
        """
            Uploading file to remote, will check if the remote's parent folder is exists, if not, create one.

        Args:
            from_local_ (str or Path): local file path
            to_remote (str or Path): remote file path

        """
        to_remote_parent = Path(to_remote).parent

        if not self.remote_exists(to_remote_parent.as_posix()):
            self.__mkdir_p__(to_remote_parent.as_posix())

        self.sftp.put(from_local_, to_remote)

    def __get_file__(self, from_remote_, to_local):
        """
            Downloading file from remote.

        Args:
            from_remote_ (str or Path): remote file path
            to_local (str or Path): local file path

        """
        to_local_parent = Path(to_local).parent

        if not to_local_parent.exists():
            to_local_parent.mkdir(parents=True, exist_ok=False)
        self.sftp.get(from_remote_, to_local)

    def recursive_list_remote_files(self, remote_dir, exclude_folder=True):
        """
            Recursively visiting remote files

        Args:
            remote_dir (str or Path): remote dir to be visited.
            exclude_folder (bool): exclude folders in visited file list.

        Returns:
            directory list and file list

        """
        queue = self.sftp.listdir(remote_dir)

        item_list = []
        dir_list = []
        while len(queue) > 0:
            item = queue.pop(0)
            full_path = os.path.join(remote_dir, item)

            if self.is_dir(full_path):
                dl = self.sftp.listdir(full_path)
                queue += [os.path.join(item, sub_item) for sub_item in dl]
                dir_list.append(item)

                if exclude_folder is False:
                    item_list.append(item)
            elif exclude_folder:
                item_list.append(item)

        return dir_list, item_list

    def from_remote(self, from_remote, to_local, show_progress=False):
        """
            Copy files or directory from remote to local

        Args:
            from_remote (str or Path): remote file path or directory
            to_local (str or Path): local file path or directory
            show_progress (bool): display progress bar

        """
        if not self.remote_exists(from_remote):
            raise Exception('Remote path %s not exists' % from_remote)
            
        to_local = Path(to_local)

        if self.is_dir(from_remote):

            if not to_local.is_dir():
                if not to_local.exists():
                    to_local.mkdir(parents=True, exist_ok=False)
                else:
                    raise Exception('Local dir %s should be a directory' % to_local)

            # gathering remote files
            remote_dir_list, remote_file_list = self.recursive_list_remote_files(from_remote, exclude_folder=True)

            # create local directories
            for dir_ in remote_dir_list:
                local_dir = (to_local / dir_)
                if not local_dir.exists():
                    local_dir.mkdir(parents=True, exist_ok=False)

            for item in tqdm(remote_file_list, desc=msg('downloading', self, True), disable=not show_progress):
                self.sftp.get(os.path.join(str(from_remote), item), os.path.join(to_local, item))
            
            return True
        elif self.remote_exists(from_remote):
            self.__get_file__(from_remote, to_local)
            return True
        else:
            notice_msg('Remote path %s not exists' % from_remote)
            return False

    def __put_file__(self, from_local_, to_remote):
        """
            Uploading file to remote, will check if the remote's parent folder is exists, if not, create one.

        Args:
            from_local_ (str or Path): local file path
            to_remote (str or Path): remote file path

        """
        to_remote_parent = Path(to_remote).parent

        if not self.remote_exists(to_remote_parent.as_posix()):
            self.__mkdir_p__(to_remote_parent.as_posix())

        self.sftp.put(from_local_, to_remote)

    def __get_file__(self, from_remote_, to_local):
        """
            Downloading file from remote.

        Args:
            from_remote_ (str or Path): remote file path
            to_local (str or Path): local file path

        """
        to_local_parent = Path(to_local).parent

        if not to_local_parent.exists():
            to_local_parent.mkdir(parents=True, exist_ok=False)

        self.sftp.get(from_remote_, to_local)

    @staticmethod
    def __mkdir_p__(sftp, remote_directory):
        """ Change to this directory, recursively making new folders if needed.
            Returns True if any folders were created.
        """
        if remote_directory == '/':
            # absolute path so change directory to root
            sftp.chdir('/')
            return
        if remote_directory == '':
            # top-level relative directory must exist
            return
        try:
            # sub-directory exists
            sftp.chdir(remote_directory)
        except IOError:
            dirname, basename = os.path.split(remote_directory.rstrip('/'))
            SFTPClient.__mkdir_p__(sftp, dirname)              # make parent directories
            sftp.mkdir(basename)                           # sub-directory missing, so created it
            sftp.chdir(basename)
            return True

    def make_remote_dirs(self, remote_dir):
        """ Recursively create directory on remote machine.
        """
        self.__mkdir_p__(self.sftp, remote_dir)


