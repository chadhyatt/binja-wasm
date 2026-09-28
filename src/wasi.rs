//! Prototypes for the WASI preview 1 (`wasi_snapshot_preview1`) and preview 0 (`wasi_unstable`)
//! imports, typed as their witx specifies and lowered as the WASI ABI passes them; a preview 0
//! type that differs from preview 1, itself or through a type it holds, is named `__wasi_unstable_*`

use std::collections::HashMap;
use std::num::NonZeroUsize;

use binaryninja::binary_view::BinaryView;
use binaryninja::rc::Ref;
use binaryninja::types::{
    EnumerationBuilder, FunctionParameter, MemberAccess, MemberScope, StructureBuilder,
    StructureType, Type,
};

use crate::module::{ImportInfo, Signature, ValueKind};

const SOURCE: &str = "wasi";

const POINTER: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Int {
    U8,
    U16,
    U32,
    U64,
}

impl Int {
    fn size(self) -> u64 {
        match self {
            Int::U8 => 1,
            Int::U16 => 2,
            Int::U32 => 4,
            Int::U64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    U8,
    U16,
    U32,
    U64,
    S64,
    String,
    Named(&'static str),
    Pointer(&'static Ty),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Definition {
    Alias(Ty),
    Handle,
    Enum(Int, &'static [&'static str]),
    Flags(Int, &'static [&'static str]),
    Record(&'static [(&'static str, Ty)]),
    List(Ty),
    Union(&'static str, &'static [&'static str]),
}

struct Function {
    name: &'static str,
    params: &'static [(&'static str, Ty)],
    results: Option<&'static [(&'static str, Ty)]>,
    unstable: bool,
}

const PREVIEW1: &[(&str, Definition)] = &[
    ("size", Definition::Alias(Ty::U32)),
    ("filesize", Definition::Alias(Ty::U64)),
    ("timestamp", Definition::Alias(Ty::U64)),
    (
        "clockid",
        Definition::Enum(
            Int::U32,
            &[
                "realtime",
                "monotonic",
                "process_cputime_id",
                "thread_cputime_id",
            ],
        ),
    ),
    (
        "errno",
        Definition::Enum(
            Int::U16,
            &[
                "success",
                "2big",
                "acces",
                "addrinuse",
                "addrnotavail",
                "afnosupport",
                "again",
                "already",
                "badf",
                "badmsg",
                "busy",
                "canceled",
                "child",
                "connaborted",
                "connrefused",
                "connreset",
                "deadlk",
                "destaddrreq",
                "dom",
                "dquot",
                "exist",
                "fault",
                "fbig",
                "hostunreach",
                "idrm",
                "ilseq",
                "inprogress",
                "intr",
                "inval",
                "io",
                "isconn",
                "isdir",
                "loop",
                "mfile",
                "mlink",
                "msgsize",
                "multihop",
                "nametoolong",
                "netdown",
                "netreset",
                "netunreach",
                "nfile",
                "nobufs",
                "nodev",
                "noent",
                "noexec",
                "nolck",
                "nolink",
                "nomem",
                "nomsg",
                "noprotoopt",
                "nospc",
                "nosys",
                "notconn",
                "notdir",
                "notempty",
                "notrecoverable",
                "notsock",
                "notsup",
                "notty",
                "nxio",
                "overflow",
                "ownerdead",
                "perm",
                "pipe",
                "proto",
                "protonosupport",
                "prototype",
                "range",
                "rofs",
                "spipe",
                "srch",
                "stale",
                "timedout",
                "txtbsy",
                "xdev",
                "notcapable",
            ],
        ),
    ),
    (
        "rights",
        Definition::Flags(
            Int::U64,
            &[
                "fd_datasync",
                "fd_read",
                "fd_seek",
                "fd_fdstat_set_flags",
                "fd_sync",
                "fd_tell",
                "fd_write",
                "fd_advise",
                "fd_allocate",
                "path_create_directory",
                "path_create_file",
                "path_link_source",
                "path_link_target",
                "path_open",
                "fd_readdir",
                "path_readlink",
                "path_rename_source",
                "path_rename_target",
                "path_filestat_get",
                "path_filestat_set_size",
                "path_filestat_set_times",
                "fd_filestat_get",
                "fd_filestat_set_size",
                "fd_filestat_set_times",
                "path_symlink",
                "path_remove_directory",
                "path_unlink_file",
                "poll_fd_readwrite",
                "sock_shutdown",
                "sock_accept",
            ],
        ),
    ),
    ("fd", Definition::Handle),
    (
        "iovec",
        Definition::Record(&[
            ("buf", Ty::Pointer(&Ty::U8)),
            ("buf_len", Ty::Named("size")),
        ]),
    ),
    (
        "ciovec",
        Definition::Record(&[
            ("buf", Ty::Pointer(&Ty::U8)),
            ("buf_len", Ty::Named("size")),
        ]),
    ),
    ("iovec_array", Definition::List(Ty::Named("iovec"))),
    ("ciovec_array", Definition::List(Ty::Named("ciovec"))),
    ("filedelta", Definition::Alias(Ty::S64)),
    ("whence", Definition::Enum(Int::U8, &["set", "cur", "end"])),
    ("dircookie", Definition::Alias(Ty::U64)),
    ("dirnamlen", Definition::Alias(Ty::U32)),
    ("inode", Definition::Alias(Ty::U64)),
    (
        "filetype",
        Definition::Enum(
            Int::U8,
            &[
                "unknown",
                "block_device",
                "character_device",
                "directory",
                "regular_file",
                "socket_dgram",
                "socket_stream",
                "symbolic_link",
            ],
        ),
    ),
    (
        "dirent",
        Definition::Record(&[
            ("d_next", Ty::Named("dircookie")),
            ("d_ino", Ty::Named("inode")),
            ("d_namlen", Ty::Named("dirnamlen")),
            ("d_type", Ty::Named("filetype")),
        ]),
    ),
    (
        "advice",
        Definition::Enum(
            Int::U8,
            &[
                "normal",
                "sequential",
                "random",
                "willneed",
                "dontneed",
                "noreuse",
            ],
        ),
    ),
    (
        "fdflags",
        Definition::Flags(Int::U16, &["append", "dsync", "nonblock", "rsync", "sync"]),
    ),
    (
        "fdstat",
        Definition::Record(&[
            ("fs_filetype", Ty::Named("filetype")),
            ("fs_flags", Ty::Named("fdflags")),
            ("fs_rights_base", Ty::Named("rights")),
            ("fs_rights_inheriting", Ty::Named("rights")),
        ]),
    ),
    ("device", Definition::Alias(Ty::U64)),
    (
        "fstflags",
        Definition::Flags(Int::U16, &["atim", "atim_now", "mtim", "mtim_now"]),
    ),
    (
        "lookupflags",
        Definition::Flags(Int::U32, &["symlink_follow"]),
    ),
    (
        "oflags",
        Definition::Flags(Int::U16, &["creat", "directory", "excl", "trunc"]),
    ),
    ("linkcount", Definition::Alias(Ty::U64)),
    (
        "filestat",
        Definition::Record(&[
            ("dev", Ty::Named("device")),
            ("ino", Ty::Named("inode")),
            ("filetype", Ty::Named("filetype")),
            ("nlink", Ty::Named("linkcount")),
            ("size", Ty::Named("filesize")),
            ("atim", Ty::Named("timestamp")),
            ("mtim", Ty::Named("timestamp")),
            ("ctim", Ty::Named("timestamp")),
        ]),
    ),
    ("userdata", Definition::Alias(Ty::U64)),
    (
        "eventtype",
        Definition::Enum(Int::U8, &["clock", "fd_read", "fd_write"]),
    ),
    (
        "eventrwflags",
        Definition::Flags(Int::U16, &["fd_readwrite_hangup"]),
    ),
    (
        "event_fd_readwrite",
        Definition::Record(&[
            ("nbytes", Ty::Named("filesize")),
            ("flags", Ty::Named("eventrwflags")),
        ]),
    ),
    (
        "event",
        Definition::Record(&[
            ("userdata", Ty::Named("userdata")),
            ("error", Ty::Named("errno")),
            ("type", Ty::Named("eventtype")),
            ("fd_readwrite", Ty::Named("event_fd_readwrite")),
        ]),
    ),
    (
        "subclockflags",
        Definition::Flags(Int::U16, &["subscription_clock_abstime"]),
    ),
    (
        "subscription_clock",
        Definition::Record(&[
            ("id", Ty::Named("clockid")),
            ("timeout", Ty::Named("timestamp")),
            ("precision", Ty::Named("timestamp")),
            ("flags", Ty::Named("subclockflags")),
        ]),
    ),
    (
        "subscription_fd_readwrite",
        Definition::Record(&[("file_descriptor", Ty::Named("fd"))]),
    ),
    (
        "subscription_u",
        Definition::Union(
            "eventtype",
            &[
                "subscription_clock",
                "subscription_fd_readwrite",
                "subscription_fd_readwrite",
            ],
        ),
    ),
    (
        "subscription",
        Definition::Record(&[
            ("userdata", Ty::Named("userdata")),
            ("u", Ty::Named("subscription_u")),
        ]),
    ),
    ("exitcode", Definition::Alias(Ty::U32)),
    (
        "signal",
        Definition::Enum(
            Int::U8,
            &[
                "none", "hup", "int", "quit", "ill", "trap", "abrt", "bus", "fpe", "kill", "usr1",
                "segv", "usr2", "pipe", "alrm", "term", "chld", "cont", "stop", "tstp", "ttin",
                "ttou", "urg", "xcpu", "xfsz", "vtalrm", "prof", "winch", "poll", "pwr", "sys",
            ],
        ),
    ),
    (
        "riflags",
        Definition::Flags(Int::U16, &["recv_peek", "recv_waitall"]),
    ),
    (
        "roflags",
        Definition::Flags(Int::U16, &["recv_data_truncated"]),
    ),
    ("siflags", Definition::Alias(Ty::U16)),
    ("sdflags", Definition::Flags(Int::U8, &["rd", "wr"])),
    ("preopentype", Definition::Enum(Int::U8, &["dir"])),
    (
        "prestat_dir",
        Definition::Record(&[("pr_name_len", Ty::Named("size"))]),
    ),
    (
        "prestat",
        Definition::Union("preopentype", &["prestat_dir"]),
    ),
];

const UNSTABLE: &[(&str, Definition)] = &[
    (
        "rights",
        Definition::Flags(
            Int::U64,
            &[
                "fd_datasync",
                "fd_read",
                "fd_seek",
                "fd_fdstat_set_flags",
                "fd_sync",
                "fd_tell",
                "fd_write",
                "fd_advise",
                "fd_allocate",
                "path_create_directory",
                "path_create_file",
                "path_link_source",
                "path_link_target",
                "path_open",
                "fd_readdir",
                "path_readlink",
                "path_rename_source",
                "path_rename_target",
                "path_filestat_get",
                "path_filestat_set_size",
                "path_filestat_set_times",
                "fd_filestat_get",
                "fd_filestat_set_size",
                "fd_filestat_set_times",
                "path_symlink",
                "path_remove_directory",
                "path_unlink_file",
                "poll_fd_readwrite",
                "sock_shutdown",
            ],
        ),
    ),
    ("whence", Definition::Enum(Int::U8, &["cur", "end", "set"])),
    ("linkcount", Definition::Alias(Ty::U32)),
    (
        "subscription_clock",
        Definition::Record(&[
            ("identifier", Ty::Named("userdata")),
            ("id", Ty::Named("clockid")),
            ("timeout", Ty::Named("timestamp")),
            ("precision", Ty::Named("timestamp")),
            ("flags", Ty::Named("subclockflags")),
        ]),
    ),
];

const FUNCTIONS: &[Function] = &[
    Function {
        name: "args_get",
        params: &[
            ("argv", Ty::Pointer(&Ty::Pointer(&Ty::U8))),
            ("argv_buf", Ty::Pointer(&Ty::U8)),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "args_sizes_get",
        params: &[],
        results: Some(&[
            ("argc", Ty::Named("size")),
            ("argv_buf_size", Ty::Named("size")),
        ]),
        unstable: true,
    },
    Function {
        name: "environ_get",
        params: &[
            ("environ", Ty::Pointer(&Ty::Pointer(&Ty::U8))),
            ("environ_buf", Ty::Pointer(&Ty::U8)),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "environ_sizes_get",
        params: &[],
        results: Some(&[
            ("environc", Ty::Named("size")),
            ("environ_buf_size", Ty::Named("size")),
        ]),
        unstable: true,
    },
    Function {
        name: "clock_res_get",
        params: &[("id", Ty::Named("clockid"))],
        results: Some(&[("resolution", Ty::Named("timestamp"))]),
        unstable: true,
    },
    Function {
        name: "clock_time_get",
        params: &[
            ("id", Ty::Named("clockid")),
            ("precision", Ty::Named("timestamp")),
        ],
        results: Some(&[("time", Ty::Named("timestamp"))]),
        unstable: true,
    },
    Function {
        name: "fd_advise",
        params: &[
            ("fd", Ty::Named("fd")),
            ("offset", Ty::Named("filesize")),
            ("len", Ty::Named("filesize")),
            ("advice", Ty::Named("advice")),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_allocate",
        params: &[
            ("fd", Ty::Named("fd")),
            ("offset", Ty::Named("filesize")),
            ("len", Ty::Named("filesize")),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_close",
        params: &[("fd", Ty::Named("fd"))],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_datasync",
        params: &[("fd", Ty::Named("fd"))],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_fdstat_get",
        params: &[("fd", Ty::Named("fd"))],
        results: Some(&[("stat", Ty::Named("fdstat"))]),
        unstable: true,
    },
    Function {
        name: "fd_fdstat_set_flags",
        params: &[("fd", Ty::Named("fd")), ("flags", Ty::Named("fdflags"))],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_fdstat_set_rights",
        params: &[
            ("fd", Ty::Named("fd")),
            ("fs_rights_base", Ty::Named("rights")),
            ("fs_rights_inheriting", Ty::Named("rights")),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_filestat_get",
        params: &[("fd", Ty::Named("fd"))],
        results: Some(&[("buf", Ty::Named("filestat"))]),
        unstable: true,
    },
    Function {
        name: "fd_filestat_set_size",
        params: &[("fd", Ty::Named("fd")), ("size", Ty::Named("filesize"))],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_filestat_set_times",
        params: &[
            ("fd", Ty::Named("fd")),
            ("atim", Ty::Named("timestamp")),
            ("mtim", Ty::Named("timestamp")),
            ("fst_flags", Ty::Named("fstflags")),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_pread",
        params: &[
            ("fd", Ty::Named("fd")),
            ("iovs", Ty::Named("iovec_array")),
            ("offset", Ty::Named("filesize")),
        ],
        results: Some(&[("nread", Ty::Named("size"))]),
        unstable: true,
    },
    Function {
        name: "fd_prestat_get",
        params: &[("fd", Ty::Named("fd"))],
        results: Some(&[("buf", Ty::Named("prestat"))]),
        unstable: true,
    },
    Function {
        name: "fd_prestat_dir_name",
        params: &[
            ("fd", Ty::Named("fd")),
            ("path", Ty::Pointer(&Ty::U8)),
            ("path_len", Ty::Named("size")),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_pwrite",
        params: &[
            ("fd", Ty::Named("fd")),
            ("iovs", Ty::Named("ciovec_array")),
            ("offset", Ty::Named("filesize")),
        ],
        results: Some(&[("nwritten", Ty::Named("size"))]),
        unstable: true,
    },
    Function {
        name: "fd_read",
        params: &[("fd", Ty::Named("fd")), ("iovs", Ty::Named("iovec_array"))],
        results: Some(&[("nread", Ty::Named("size"))]),
        unstable: true,
    },
    Function {
        name: "fd_readdir",
        params: &[
            ("fd", Ty::Named("fd")),
            ("buf", Ty::Pointer(&Ty::U8)),
            ("buf_len", Ty::Named("size")),
            ("cookie", Ty::Named("dircookie")),
        ],
        results: Some(&[("bufused", Ty::Named("size"))]),
        unstable: true,
    },
    Function {
        name: "fd_renumber",
        params: &[("fd", Ty::Named("fd")), ("to", Ty::Named("fd"))],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_seek",
        params: &[
            ("fd", Ty::Named("fd")),
            ("offset", Ty::Named("filedelta")),
            ("whence", Ty::Named("whence")),
        ],
        results: Some(&[("newoffset", Ty::Named("filesize"))]),
        unstable: true,
    },
    Function {
        name: "fd_sync",
        params: &[("fd", Ty::Named("fd"))],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "fd_tell",
        params: &[("fd", Ty::Named("fd"))],
        results: Some(&[("offset", Ty::Named("filesize"))]),
        unstable: true,
    },
    Function {
        name: "fd_write",
        params: &[("fd", Ty::Named("fd")), ("iovs", Ty::Named("ciovec_array"))],
        results: Some(&[("nwritten", Ty::Named("size"))]),
        unstable: true,
    },
    Function {
        name: "path_create_directory",
        params: &[("fd", Ty::Named("fd")), ("path", Ty::String)],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "path_filestat_get",
        params: &[
            ("fd", Ty::Named("fd")),
            ("flags", Ty::Named("lookupflags")),
            ("path", Ty::String),
        ],
        results: Some(&[("buf", Ty::Named("filestat"))]),
        unstable: true,
    },
    Function {
        name: "path_filestat_set_times",
        params: &[
            ("fd", Ty::Named("fd")),
            ("flags", Ty::Named("lookupflags")),
            ("path", Ty::String),
            ("atim", Ty::Named("timestamp")),
            ("mtim", Ty::Named("timestamp")),
            ("fst_flags", Ty::Named("fstflags")),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "path_link",
        params: &[
            ("old_fd", Ty::Named("fd")),
            ("old_flags", Ty::Named("lookupflags")),
            ("old_path", Ty::String),
            ("new_fd", Ty::Named("fd")),
            ("new_path", Ty::String),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "path_open",
        params: &[
            ("fd", Ty::Named("fd")),
            ("dirflags", Ty::Named("lookupflags")),
            ("path", Ty::String),
            ("oflags", Ty::Named("oflags")),
            ("fs_rights_base", Ty::Named("rights")),
            ("fs_rights_inheriting", Ty::Named("rights")),
            ("fdflags", Ty::Named("fdflags")),
        ],
        results: Some(&[("opened_fd", Ty::Named("fd"))]),
        unstable: true,
    },
    Function {
        name: "path_readlink",
        params: &[
            ("fd", Ty::Named("fd")),
            ("path", Ty::String),
            ("buf", Ty::Pointer(&Ty::U8)),
            ("buf_len", Ty::Named("size")),
        ],
        results: Some(&[("bufused", Ty::Named("size"))]),
        unstable: true,
    },
    Function {
        name: "path_remove_directory",
        params: &[("fd", Ty::Named("fd")), ("path", Ty::String)],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "path_rename",
        params: &[
            ("fd", Ty::Named("fd")),
            ("old_path", Ty::String),
            ("new_fd", Ty::Named("fd")),
            ("new_path", Ty::String),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "path_symlink",
        params: &[
            ("old_path", Ty::String),
            ("fd", Ty::Named("fd")),
            ("new_path", Ty::String),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "path_unlink_file",
        params: &[("fd", Ty::Named("fd")), ("path", Ty::String)],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "poll_oneoff",
        params: &[
            ("in", Ty::Pointer(&Ty::Named("subscription"))),
            ("out", Ty::Pointer(&Ty::Named("event"))),
            ("nsubscriptions", Ty::Named("size")),
        ],
        results: Some(&[("nevents", Ty::Named("size"))]),
        unstable: true,
    },
    Function {
        name: "proc_exit",
        params: &[("rval", Ty::Named("exitcode"))],
        results: None,
        unstable: true,
    },
    Function {
        name: "proc_raise",
        params: &[("sig", Ty::Named("signal"))],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "sched_yield",
        params: &[],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "random_get",
        params: &[
            ("buf", Ty::Pointer(&Ty::U8)),
            ("buf_len", Ty::Named("size")),
        ],
        results: Some(&[]),
        unstable: true,
    },
    Function {
        name: "sock_accept",
        params: &[("fd", Ty::Named("fd")), ("flags", Ty::Named("fdflags"))],
        results: Some(&[("retptr0", Ty::Named("fd"))]),
        unstable: false,
    },
    Function {
        name: "sock_recv",
        params: &[
            ("fd", Ty::Named("fd")),
            ("ri_data", Ty::Named("iovec_array")),
            ("ri_flags", Ty::Named("riflags")),
        ],
        results: Some(&[
            ("ro_datalen", Ty::Named("size")),
            ("ro_flags", Ty::Named("roflags")),
        ]),
        unstable: true,
    },
    Function {
        name: "sock_send",
        params: &[
            ("fd", Ty::Named("fd")),
            ("si_data", Ty::Named("ciovec_array")),
            ("si_flags", Ty::Named("siflags")),
        ],
        results: Some(&[("so_datalen", Ty::Named("size"))]),
        unstable: true,
    },
    Function {
        name: "sock_shutdown",
        params: &[("fd", Ty::Named("fd")), ("how", Ty::Named("sdflags"))],
        results: Some(&[]),
        unstable: true,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    size: u64,
    align: u64,
}

impl Layout {
    fn scalar(size: u64) -> Self {
        Self { size, align: size }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tagged {
    payload: Layout,
    at: u64,
    whole: Layout,
}

fn referenced(ty: Ty) -> Option<&'static str> {
    match ty {
        Ty::Named(name) => Some(name),
        Ty::Pointer(to) => referenced(*to),
        Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64 | Ty::S64 | Ty::String => None,
    }
}

fn held(definition: Definition) -> Vec<&'static str> {
    match definition {
        Definition::Alias(ty) | Definition::List(ty) => referenced(ty).into_iter().collect(),
        Definition::Record(fields) => fields
            .iter()
            .filter_map(|(_, ty)| referenced(*ty))
            .collect(),
        Definition::Union(tag, cases) => {
            std::iter::once(tag).chain(cases.iter().copied()).collect()
        }
        Definition::Handle | Definition::Enum(..) | Definition::Flags(..) => Vec::new(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Version {
    Preview1,
    Unstable,
}

impl Version {
    fn of(module: &str) -> Option<Self> {
        match module {
            "wasi_snapshot_preview1" => Some(Self::Preview1),
            "wasi_unstable" => Some(Self::Unstable),
            _ => None,
        }
    }

    fn definition(self, name: &str) -> Option<Definition> {
        let find = |table: &[(&str, Definition)]| {
            table
                .iter()
                .find(|(defined, _)| *defined == name)
                .map(|(_, definition)| *definition)
        };
        match self {
            Self::Preview1 => find(PREVIEW1),
            Self::Unstable => find(UNSTABLE).or_else(|| find(PREVIEW1)),
        }
    }

    fn function(self, name: &str) -> Option<&'static Function> {
        FUNCTIONS
            .iter()
            .find(|function| function.name == name && (self == Self::Preview1 || function.unstable))
    }

    fn revised(self, name: &str) -> bool {
        self == Self::Unstable
            && (UNSTABLE.iter().any(|(defined, _)| *defined == name)
                || self.definition(name).is_some_and(|definition| {
                    held(definition).into_iter().any(|held| self.revised(held))
                }))
    }

    fn type_name(self, name: &str, part: &str) -> String {
        let prefix = if self.revised(name) {
            "__wasi_unstable"
        } else {
            "__wasi"
        };
        format!("{prefix}_{name}{part}_t")
    }

    fn layout(self, ty: Ty) -> Option<Layout> {
        match ty {
            Ty::U8 => Some(Layout::scalar(1)),
            Ty::U16 => Some(Layout::scalar(2)),
            Ty::U32 => Some(Layout::scalar(4)),
            Ty::U64 | Ty::S64 => Some(Layout::scalar(8)),
            Ty::Pointer(_) => Some(Layout::scalar(POINTER)),
            Ty::String => None,
            Ty::Named(name) => match self.definition(name)? {
                Definition::Alias(ty) => self.layout(ty),
                Definition::Handle => Some(Layout::scalar(4)),
                Definition::Enum(int, _) | Definition::Flags(int, _) => {
                    Some(Layout::scalar(int.size()))
                }
                Definition::Record(fields) => self.record(fields).map(|(_, layout)| layout),
                Definition::Union(tag, cases) => self.tagged(tag, cases).map(|tagged| tagged.whole),
                Definition::List(_) => None,
            },
        }
    }

    fn record(self, fields: &[(&str, Ty)]) -> Option<(Vec<u64>, Layout)> {
        let mut offsets = Vec::with_capacity(fields.len());
        let (mut end, mut align) = (0u64, 1u64);
        for (_, ty) in fields {
            let field = self.layout(*ty)?;
            let at = end.next_multiple_of(field.align);
            offsets.push(at);
            end = at + field.size;
            align = align.max(field.align);
        }
        let size = end.next_multiple_of(align);
        Some((offsets, Layout { size, align }))
    }

    fn tagged(self, tag: &'static str, cases: &[&'static str]) -> Option<Tagged> {
        let tag = self.layout(Ty::Named(tag))?;
        let mut payload = Layout { size: 0, align: 1 };
        for case in cases {
            let case = self.layout(Ty::Named(case))?;
            payload.size = payload.size.max(case.size);
            payload.align = payload.align.max(case.align);
        }
        payload.size = payload.size.next_multiple_of(payload.align);
        let at = tag.size.next_multiple_of(payload.align);
        let align = tag.align.max(payload.align);
        let whole = Layout {
            size: (at + payload.size).next_multiple_of(align),
            align,
        };
        Some(Tagged { payload, at, whole })
    }

    fn lowered(self, ty: Ty) -> Option<Vec<ValueKind>> {
        let kinds = match ty {
            Ty::U8 | Ty::U16 | Ty::U32 | Ty::Pointer(_) => vec![ValueKind::I32],
            Ty::U64 | Ty::S64 => vec![ValueKind::I64],
            Ty::String => vec![ValueKind::I32; 2],
            Ty::Named(name) => match self.definition(name)? {
                Definition::Alias(ty) => return self.lowered(ty),
                Definition::Handle => vec![ValueKind::I32],
                Definition::Enum(Int::U64, _) | Definition::Flags(Int::U64, _) => {
                    vec![ValueKind::I64]
                }
                Definition::Enum(..) | Definition::Flags(..) => vec![ValueKind::I32],
                Definition::List(_) => vec![ValueKind::I32; 2],
                Definition::Record(_) | Definition::Union(..) => return None,
            },
        };
        Some(kinds)
    }

    fn signature(self, function: &Function) -> Option<Signature> {
        let mut params = Vec::new();
        for (_, ty) in function.params {
            params.extend(self.lowered(*ty)?);
        }
        let results = function.results.unwrap_or_default();
        params.extend(results.iter().map(|_| ValueKind::I32));
        let results = match function.results {
            Some(_) => vec![ValueKind::I32],
            None => Vec::new(),
        };
        Some(Signature { params, results })
    }
}

pub struct Import {
    pub ty: Ref<Type>,
    pub returns: bool,
}

fn specified(import: &ImportInfo) -> Option<(Version, &'static Function)> {
    specified_as(
        Version::of(&import.module)?,
        &import.field,
        &import.signature,
    )
}

fn specified_as(
    version: Version,
    name: &str,
    signature: &Signature,
) -> Option<(Version, &'static Function)> {
    let function = version.function(name)?;
    (version.signature(function)? == *signature).then_some((version, function))
}

pub fn specifies(import: &ImportInfo) -> bool {
    specified(import).is_some()
}

pub fn prototype(view: &BinaryView, import: &ImportInfo) -> Option<Import> {
    let (version, function) = specified(import)?;
    build(view, version, function)
}

pub fn implemented_as(view: &BinaryView, name: &str, signature: &Signature) -> Option<Import> {
    let (version, function) = specified_as(Version::Preview1, name, signature)?;
    build(view, version, function)
}

fn build(view: &BinaryView, version: Version, function: &Function) -> Option<Import> {
    let mut types = Types {
        view,
        version,
        defined: HashMap::new(),
    };
    let mut parameters = Vec::new();
    for (name, ty) in function.params {
        parameters.extend(types.parameters(name, *ty)?);
    }
    for (name, ty) in function.results.unwrap_or_default() {
        let ty = types.ty(*ty)?;
        parameters.push(FunctionParameter::new(
            pointer(&ty),
            (*name).to_owned(),
            None,
        ));
    }
    let returned = match function.results {
        Some(_) => types.named("errno")?,
        None => Type::void(),
    };
    Some(Import {
        ty: Type::function(&returned, parameters, false),
        returns: function.results.is_some(),
    })
}

fn type_id(name: &str) -> String {
    format!("{SOURCE}:{name}")
}

pub fn give_way(view: &BinaryView, name: &str) {
    let id = type_id(name);
    if view.type_name_by_id(&id).is_some() {
        view.undefine_auto_type(&id);
    }
}

fn pointer(to: &Type) -> Ref<Type> {
    Type::pointer_of_width(to, POINTER as usize, false, false, None)
}

fn enumeration(
    name: &str,
    int: Int,
    members: impl Iterator<Item = (&'static str, u64)>,
) -> Option<Ref<Type>> {
    let mut builder = EnumerationBuilder::new();
    for (member, value) in members {
        let member = format!("__WASI_{name}_{member}").to_uppercase();
        builder.insert(&member, value);
    }
    let width = NonZeroUsize::new(usize::try_from(int.size()).ok()?)?;
    Some(Type::enumeration(&builder.finalize(), width, false))
}

struct Types<'a> {
    view: &'a BinaryView,
    version: Version,
    defined: HashMap<&'static str, Ref<Type>>,
}

impl Types<'_> {
    fn ty(&mut self, ty: Ty) -> Option<Ref<Type>> {
        Some(match ty {
            Ty::U8 => Type::int(1, false),
            Ty::U16 => Type::int(2, false),
            Ty::U32 => Type::int(4, false),
            Ty::U64 => Type::int(8, false),
            Ty::S64 => Type::int(8, true),
            Ty::Pointer(to) => {
                let to = self.ty(*to)?;
                pointer(&to)
            }
            Ty::Named(name) => self.named(name)?,
            Ty::String => return None,
        })
    }

    fn parameters(&mut self, name: &str, ty: Ty) -> Option<Vec<FunctionParameter>> {
        let element = match ty {
            Ty::String => Some(Type::char()),
            Ty::Named(list) => match self.version.definition(list)? {
                Definition::List(element) => Some(self.ty(element)?),
                _ => None,
            },
            _ => None,
        };
        Some(match element {
            Some(element) => vec![
                FunctionParameter::new(pointer(&element), name.to_owned(), None),
                FunctionParameter::new(self.named("size")?, format!("{name}_len"), None),
            ],
            None => vec![FunctionParameter::new(self.ty(ty)?, name.to_owned(), None)],
        })
    }

    fn named(&mut self, name: &'static str) -> Option<Ref<Type>> {
        if let Some(defined) = self.defined.get(name) {
            return Some(defined.clone());
        }
        let body = self.body(name)?;
        let defined = self.define(self.version.type_name(name, ""), &body);
        self.defined.insert(name, defined.clone());
        Some(defined)
    }

    fn define(&self, name: String, body: &Type) -> Ref<Type> {
        let registered = self
            .view
            .define_auto_type_with_id(name.as_str(), &type_id(&name), body);
        Type::named_type_from_type(registered, body)
    }

    fn body(&mut self, name: &'static str) -> Option<Ref<Type>> {
        let version = self.version;
        match version.definition(name)? {
            Definition::Alias(ty) => self.ty(ty),
            Definition::Handle => Some(Type::int(4, true)),
            Definition::Enum(int, cases) => enumeration(name, int, cases.iter().copied().zip(0..)),
            Definition::Flags(int, bits) => enumeration(
                name,
                int,
                bits.iter().copied().zip((0..).map(|bit| 1u64 << bit)),
            ),
            Definition::Record(fields) => {
                let (offsets, layout) = version.record(fields)?;
                let mut builder = StructureBuilder::new();
                for ((field, ty), at) in fields.iter().zip(offsets) {
                    let ty = self.ty(*ty)?;
                    builder.insert(
                        &ty,
                        field,
                        at,
                        false,
                        MemberAccess::PublicAccess,
                        MemberScope::NoScope,
                    );
                }
                builder
                    .width(layout.size)
                    .alignment(usize::try_from(layout.align).ok()?);
                Some(Type::structure(&builder.finalize()))
            }
            Definition::Union(tag, cases) => {
                let tagged = version.tagged(tag, cases)?;
                let Definition::Enum(_, members) = version.definition(tag)? else {
                    return None;
                };
                let mut union = StructureBuilder::new();
                union.structure_type(StructureType::UnionStructureType);
                for (member, case) in members.iter().zip(cases) {
                    let case = self.named(case)?;
                    union.insert(
                        &case,
                        member,
                        0,
                        false,
                        MemberAccess::PublicAccess,
                        MemberScope::NoScope,
                    );
                }
                union
                    .width(tagged.payload.size)
                    .alignment(usize::try_from(tagged.payload.align).ok()?);
                let union = self.define(
                    version.type_name(name, "_u"),
                    &Type::structure(&union.finalize()),
                );
                let tag = self.named(tag)?;
                let mut outer = StructureBuilder::new();
                outer.insert(
                    &tag,
                    "tag",
                    0,
                    false,
                    MemberAccess::PublicAccess,
                    MemberScope::NoScope,
                );
                outer.insert(
                    &union,
                    "u",
                    tagged.at,
                    false,
                    MemberAccess::PublicAccess,
                    MemberScope::NoScope,
                );
                outer
                    .width(tagged.whole.size)
                    .alignment(usize::try_from(tagged.whole.align).ok()?);
                Some(Type::structure(&outer.finalize()))
            }
            Definition::List(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<ValueKind> {
        text.chars()
            .map(|kind| match kind {
                'i' => ValueKind::I32,
                'I' => ValueKind::I64,
                other => panic!("no kind {other}"),
            })
            .collect()
    }

    #[test]
    fn every_function_lowers_to_the_signature_wasi_libc_imports_it_with() {
        let imported = [
            ("args_get", "ii"),
            ("args_sizes_get", "ii"),
            ("environ_get", "ii"),
            ("environ_sizes_get", "ii"),
            ("clock_res_get", "ii"),
            ("clock_time_get", "iIi"),
            ("fd_advise", "iIIi"),
            ("fd_allocate", "iII"),
            ("fd_close", "i"),
            ("fd_datasync", "i"),
            ("fd_fdstat_get", "ii"),
            ("fd_fdstat_set_flags", "ii"),
            ("fd_fdstat_set_rights", "iII"),
            ("fd_filestat_get", "ii"),
            ("fd_filestat_set_size", "iI"),
            ("fd_filestat_set_times", "iIIi"),
            ("fd_pread", "iiiIi"),
            ("fd_prestat_get", "ii"),
            ("fd_prestat_dir_name", "iii"),
            ("fd_pwrite", "iiiIi"),
            ("fd_read", "iiii"),
            ("fd_readdir", "iiiIi"),
            ("fd_renumber", "ii"),
            ("fd_seek", "iIii"),
            ("fd_sync", "i"),
            ("fd_tell", "ii"),
            ("fd_write", "iiii"),
            ("path_create_directory", "iii"),
            ("path_filestat_get", "iiiii"),
            ("path_filestat_set_times", "iiiiIIi"),
            ("path_link", "iiiiiii"),
            ("path_open", "iiiiiIIii"),
            ("path_readlink", "iiiiii"),
            ("path_remove_directory", "iii"),
            ("path_rename", "iiiiii"),
            ("path_symlink", "iiiii"),
            ("path_unlink_file", "iii"),
            ("poll_oneoff", "iiii"),
            ("sched_yield", ""),
            ("random_get", "ii"),
            ("sock_accept", "iii"),
            ("sock_recv", "iiiiii"),
            ("sock_send", "iiiii"),
            ("sock_shutdown", "ii"),
        ];
        for (name, params) in imported {
            let function = Version::Preview1.function(name).expect("specified");
            assert_eq!(
                Version::Preview1.signature(function),
                Some(Signature {
                    params: kinds(params),
                    results: kinds("i"),
                }),
                "{name}"
            );
        }
        let exit = Version::Preview1.function("proc_exit").expect("specified");
        assert_eq!(
            Version::Preview1.signature(exit),
            Some(Signature {
                params: kinds("i"),
                results: Vec::new(),
            }),
            "proc_exit, which never returns"
        );
        let raise = Version::Preview1.function("proc_raise").expect("specified");
        assert_eq!(
            Version::Preview1.signature(raise),
            Some(Signature {
                params: kinds("i"),
                results: kinds("i"),
            }),
            "proc_raise, which wasi-libc no longer imports"
        );
        assert_eq!(FUNCTIONS.len(), imported.len() + 2);
    }

    #[test]
    fn a_record_is_laid_out_as_wasi_libc_asserts_it_is() {
        let record = |version: Version, name: &str| {
            let Some(Definition::Record(fields)) = version.definition(name) else {
                panic!("{name} is no record");
            };
            let (offsets, layout) = version.record(fields).expect("laid out");
            (offsets, layout.size, layout.align)
        };
        let expected: [(&str, &[u64], u64, u64); 11] = [
            ("iovec", &[0, 4], 8, 4),
            ("ciovec", &[0, 4], 8, 4),
            ("dirent", &[0, 8, 16, 20], 24, 8),
            ("fdstat", &[0, 2, 8, 16], 24, 8),
            ("filestat", &[0, 8, 16, 24, 32, 40, 48, 56], 64, 8),
            ("event_fd_readwrite", &[0, 8], 16, 8),
            ("event", &[0, 8, 10, 16], 32, 8),
            ("subscription_clock", &[0, 8, 16, 24], 32, 8),
            ("subscription_fd_readwrite", &[0], 4, 4),
            ("subscription", &[0, 8], 48, 8),
            ("prestat_dir", &[0], 4, 4),
        ];
        for (name, offsets, size, align) in expected {
            assert_eq!(
                record(Version::Preview1, name),
                (offsets.to_vec(), size, align),
                "{name}"
            );
        }
        assert_eq!(
            Version::Preview1.layout(Ty::Named("subscription_u")),
            Some(Layout { size: 40, align: 8 })
        );
        assert_eq!(
            Version::Preview1.layout(Ty::Named("prestat")),
            Some(Layout { size: 8, align: 4 })
        );

        assert_eq!(
            record(Version::Unstable, "filestat"),
            (vec![0, 8, 16, 20, 24, 32, 40, 48], 56, 8),
            "a link count half as wide"
        );
        assert_eq!(
            record(Version::Unstable, "subscription"),
            (vec![0, 8], 56, 8),
            "a clock subscription that carries its own identifier"
        );
    }

    #[test]
    fn every_type_a_function_names_is_defined_and_laid_out() {
        for version in [Version::Preview1, Version::Unstable] {
            for function in FUNCTIONS
                .iter()
                .filter(|function| version == Version::Preview1 || function.unstable)
            {
                assert!(version.signature(function).is_some(), "{}", function.name);
                for (_, ty) in function.results.unwrap_or_default() {
                    assert!(version.layout(*ty).is_some(), "{}", function.name);
                }
            }
            for (name, _) in PREVIEW1 {
                let lowered = version.lowered(Ty::Named(name)).is_some();
                let laid_out = version.layout(Ty::Named(name)).is_some();
                assert!(lowered || laid_out, "{name}");
            }
        }
    }

    #[test]
    fn a_preview_0_type_is_named_apart_only_where_it_differs() {
        let names = |name: &str| {
            [Version::Preview1, Version::Unstable].map(|version| version.type_name(name, ""))
        };
        assert_eq!(names("fd"), ["__wasi_fd_t", "__wasi_fd_t"]);
        assert_eq!(names("event"), ["__wasi_event_t", "__wasi_event_t"]);
        assert_eq!(
            names("whence"),
            ["__wasi_whence_t", "__wasi_unstable_whence_t"],
            "the same cases in another order"
        );
        assert_eq!(
            names("fdstat"),
            ["__wasi_fdstat_t", "__wasi_unstable_fdstat_t"],
            "rights without the one preview 1 added"
        );
        assert_eq!(
            names("subscription_u"),
            [
                "__wasi_subscription_u_t",
                "__wasi_unstable_subscription_u_t"
            ],
            "a union holding a revised record"
        );
        assert_eq!(
            Version::Unstable.type_name("subscription_u", "_u"),
            "__wasi_unstable_subscription_u_u_t"
        );
    }

    #[test]
    fn an_import_is_specified_only_with_the_signature_its_witx_lowers_to() {
        let import = |module: &str, field: &str, params: &str| ImportInfo {
            module: module.to_owned(),
            field: field.to_owned(),
            signature: Signature {
                params: kinds(params),
                results: kinds("i"),
            },
        };
        assert!(specifies(&import(
            "wasi_snapshot_preview1",
            "fd_write",
            "iiii"
        )));
        assert!(specifies(&import("wasi_unstable", "fd_write", "iiii")));
        assert!(
            !specifies(&import("wasi_snapshot_preview1", "fd_write", "IIII")),
            "pointers as wide as memory64 would make them"
        );
        assert!(!specifies(&import("env", "fd_write", "iiii")));
        assert!(!specifies(&import("wasi_unstable", "sock_accept", "iii")));
    }

    #[test]
    fn only_the_wasi_modules_are_known_and_only_with_the_functions_they_have() {
        assert_eq!(
            Version::of("wasi_snapshot_preview1"),
            Some(Version::Preview1)
        );
        assert_eq!(Version::of("wasi_unstable"), Some(Version::Unstable));
        assert_eq!(Version::of("env"), None);
        assert!(Version::Preview1.function("sock_accept").is_some());
        assert!(Version::Unstable.function("sock_accept").is_none());
        assert!(Version::Unstable.function("fd_write").is_some());
        assert!(Version::Preview1.function("fd_open").is_none());
    }
}
