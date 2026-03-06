import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#d45f1f',
      light: '#f29f5a',
      dark: '#8f3f13'
    },
    secondary: {
      main: '#0d5e7a',
      light: '#3f8eaa',
      dark: '#083b4d'
    },
    success: {
      main: '#1f7a58'
    },
    error: {
      main: '#ad2e3f'
    },
    warning: {
      main: '#d47b1f'
    },
    background: {
      default: '#f4f5f2',
      paper: '#ffffff'
    },
    text: {
      primary: '#111827',
      secondary: '#475569'
    },
    divider: '#d8e0e8'
  },
  shape: {
    borderRadius: 14
  },
  typography: {
    fontFamily: '"Plus Jakarta Sans", "Sora", "Avenir Next", "Segoe UI", sans-serif',
    h4: {
      fontWeight: 800,
      letterSpacing: '-0.02em',
      lineHeight: 1.15
    },
    h5: {
      fontWeight: 750,
      letterSpacing: '-0.015em',
      lineHeight: 1.18
    },
    h6: {
      fontWeight: 700,
      letterSpacing: '-0.01em'
    },
    subtitle1: {
      fontWeight: 600,
      lineHeight: 1.45
    },
    body1: {
      lineHeight: 1.6,
      letterSpacing: '0.005em'
    },
    body2: {
      lineHeight: 1.55,
      letterSpacing: '0.004em'
    },
    button: {
      textTransform: 'none',
      fontWeight: 700,
      letterSpacing: '0.01em'
    }
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '::selection': {
          backgroundColor: 'rgba(212, 95, 31, 0.22)',
          color: '#1f2937'
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          border: '1px solid #d8e0e8',
          boxShadow: '0 18px 34px rgba(15, 23, 42, 0.08)'
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          paddingInline: 14,
          minHeight: 38
        }
      }
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 999,
          fontWeight: 600,
          letterSpacing: '0.01em'
        },
        label: {
          paddingInline: 10
        }
      }
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 11
        }
      }
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          minHeight: 38
        }
      }
    },
    MuiTab: {
      styleOverrides: {
        root: {
          minHeight: 38,
          fontWeight: 700,
          letterSpacing: '0.01em'
        }
      }
    }
  }
});

export default theme;
